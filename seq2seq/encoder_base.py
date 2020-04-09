from typing import Optional, Tuple, List, Callable, Union
import os
import json
import codecs
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]


class _RNNEncoderBase(nn.Module):

    def __init__(self, stateful: bool=False) -> None:
        super(RNN, self).__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self,
                             module: Callable[
                                 [PackedSequence, Optional[RnnState]],
                                 Tuple[Union[PackedSequence, torch.Tensor], RnnState]
                             ],
                             inputs: torch.Tensor,
                             pad_token: int,
                             hidden_state: Optional[RnnState] = None):
        """input을 정렬한 후에 forward module을 돌려 output과 final hidden state를 반환"""
        # Sorting
        batch_size = mask.size(0) # batch_first=True
        max_sequence_length = inputs.size(1)
        sequence_length = (inputs != pad_token).sum(dim=-1)
        sorted_sequence_length, permutation_index = sequence_length.sort(0, descending=True)
        sorted_inputs = inputs.index_select(0, permutation_index)
        _, restoration_indices = permutation_index.sort(0, descending=False)

        # Packing
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_length.data.tolist(),
                                                     batch_first=True)

        # Prepare the initial states
        if not self.stateful:
            if hidden_state == None:
                initial_states = hidden_state # Set None
            elif isinstance(hidden_states, tuple):
                initial_states = [state.index_select(1, sorting_indices)
                                  for state in hidden_state]
            else:
                initial_states = self._get_initial_states(
                    batch_size, max_sequence_length, sorting_indices)
        else:
            initial_states = self._get_initial_states(
                batch_size, max_sequence_length, sorting_indices)

        # Actually call the module on the sorted PackedSequence
        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration_indices

    def _get_initial_states(self,
                            batch_size: int,
                            max_sequence_length: int,
                            sorting_indices: torch.LongTensor) -> Optional[RnnState]:
        """
        (a) RNN의 초기 상태를 반환
        (b) batch의 새로운 요소에 대한 초기 상태를 추가하기 위해
           메서드를 호출, 상태를 변경(mutate)하여 바뀐 batch_size를 handling
        (c) batch의 각 요소(sentence)의 sequence 길이로 state를 정렬
        (d) pad 작업 수행 후 행을 제거
        (e) 이전에 호출됐을 때보다 batch_size가 크면 상태를 변경(mutate)
            - (b)의 특별 case.

        이 메서드는
            (1) 처음 호출되어 아무 상태가 없는 경우
            (2) RNN이 heterogeneous state를 가질 때
        의 경우를 처리해야 하기 때문에 return값이 복잡함

        (1) module이 처음 호출됬을 때 ``module``의 타입이 무엇이든 ``None`` 반환
        (2) Otherwise,
            - LSTM의 경우 tuple of ``torch.Tensor``
              shape: ``(num_layers, max_seq_len, state_size)``
                 and ``(num_layers, max_seq_len, memory_size)``
            - GRU의 경우  single ``torch.Tensor``
              shape: ``(num_layers, max_seq_len, state_size)``
        """
        # (1)의 경우 처리
        if self._states is None:
            return None

        # (2)의 경우 처리
        if batch_size > self._states[0].size(1):
            # (e)의 경우 처리
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            # state의 shape은 (num_layers, batch_size, hidden_size)
            for state in self._states:
                zeros = state.data.new(state.size(0),
                                       num_states_to_concat,
                                       state.size(2)).fill_(0)
                zeros = Variable(zeros) # torch 1.4.0에선 Variable과 Tensor가 같음
                resized_states.append(torch.cat([state, zeros], dim=1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            # (b)의 경우 처리
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state in self._states)
        else:
            correctly_shaped_states = self._states

        if len(self._states) == 1:
            # (2)에서 GRU에 해당
            correctly_shaped_states = correctly_shaped_states[0] # unpack from tuple
            # (c)처리
            sorted_state = correctly_shaped_states.index_select(1, sorting_indices)
            return sorted_state
        else:
            # (2)에서 LSTM에 해당
            sorted_states = [states.index_select(1, sorting_indices)
                             for state in correctly_shaped_states]
            return tuple(sorted_states)

    def _update_states(self,
                       final_states: RnnStateStorage,
                       restoration_indices: torch.LongTensor) -> None:
        # TODO(Mark): seems weird to sort here, but append zeros in the subclasses
        # which way around is best?
        new_unsorted_states = [state.index_select(1, restoration_indices)
                               for state in final_states]

        if self._states is None:
            self._states = tuple(
                [Variable(state.data)
                 for state in new_unsorted_states]
            )
        else:
            # 어떤 상태가 RNN 계산에 사용될 지 나타내기 위해
            # (new_batch_size,) 크기의 mask를 생성
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # 사용하지 않은 state에 대한 mask (1, new_batch_size, 1)
            used_new_rows_mask = [(state[0, :, :].sum(-1) != 0.0).float().view(
                                  1, new_state_batch_size, 1)
                                  for state in new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states,
                                                           new_unsorted_states,
                                                           used_new_rows_mask):
                    # all zero인 row(사용하지 않은 state)만 살림
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    # 사용하여 업데이트할 상태 + 사용하지 않은 상태로 기존 상태를 채움
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    # 계산 그래프 분리
                    new_states.append(Variable(old_state.data))
            else:
                for old_state, new_state, used_mask in zip(self._states,
                                                           new_unsorted_states,
                                                           used_new_rows_mask):
                    # all zero인 row(사용하지 않은 state)만 살림
                    masked_old_state = old_state * (1 - used_mask)
                    # 사용하지 않았던 상태의 값들을 이번엔 new_state에 기록
                    new_state += masked_old_state
                    # 계산 그래프 분리
                    new_states.append(Variable(new_state))
            # 왜 current_state_batch_size < new_state_batch_size인 경우를 고려하지 않는가?
            # `_get_initial_state` 메서드에서 이미 처리
            self._states = tuple(new_states)

    def reset_states(self):
        self._states = None
