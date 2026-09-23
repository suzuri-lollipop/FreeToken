"""DetokenizeManager incremental state with several msgs for one uid in a batch.

The MTP spec drain emits up to two DetokenizeMsg per request per step; the
batched slice capture must not let the second msg re-emit the first one's text
(it did: both read slices were taken before either msg advanced the offsets).
"""

from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _FakeTok:
    eos_token_id = 2

    def batch_decode(self, ids_list):
        return ["".join(f"t{i} " for i in ids) for ids in ids_list]


def _msg(uid, tok, finished=False):
    return DetokenizeMsg(uid=uid, next_token=tok, finished=finished,
                         finish_reason="stop" if finished else None,
                         matched_stop=None, stop_strs=None)


def test_two_msgs_one_uid_do_not_duplicate_text():
    m = DetokenizeManager(_FakeTok())
    out = m.detokenize([_msg(1, 10), _msg(1, 11, finished=True)])
    assert "".join(out) == "t10 t11 "


def test_same_uid_split_across_batches_matches_the_single_batch_result():
    one = DetokenizeManager(_FakeTok())
    joined = "".join(one.detokenize([_msg(1, 10), _msg(1, 11, finished=True)]))
    two = DetokenizeManager(_FakeTok())
    split = "".join(two.detokenize([_msg(1, 10)]) + two.detokenize([_msg(1, 11, finished=True)]))
    assert joined == split == "t10 t11 "


def test_distinct_uids_keep_the_batched_path():
    m = DetokenizeManager(_FakeTok())
    out = m.detokenize([_msg(1, 10), _msg(2, 20), _msg(1, 11), _msg(2, 21, finished=True)])
    per_uid = {}
    for msg, s in zip([_msg(1, 10), _msg(2, 20), _msg(1, 11), _msg(2, 21)], out):
        per_uid.setdefault(msg.uid, "")
        per_uid[msg.uid] += s
    assert per_uid == {1: "t10 t11 ", 2: "t20 t21 "}
