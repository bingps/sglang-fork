"""ForwardBatch.real_batch_size: batch size excluding DP-sync padding rows."""

import unittest


class TestRealBatchSize(unittest.TestCase):
    """Guards the pre-padding request count used by the HiSparse swap-in guard.

    Bug (review c0b3d6c43 #1): _prepare_eager_forward_batch calls
    prepare_mlp_sync_batch, which rewrites forward_batch.batch_size to the
    DP-PADDED size and stashes the original in _original_batch_size, and only
    afterwards filled the coordinator's num_real_reqs from batch_size. The
    kernel guard `bid >= num_real_reqs[0]`, whose job is to drop DP padding
    blocks, therefore stopped filtering anything on an uneven-DP eager verify.
    Reading batch_size again here would turn this red.
    """

    def _fb(self, batch_size, original=None):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        # Only the two fields the property reads; ForwardBatch.__init__ needs a
        # full model runner, which this pure-logic property does not.
        fb = ForwardBatch.__new__(ForwardBatch)
        fb.batch_size = batch_size
        fb._original_batch_size = original
        return fb

    def test_unpadded_batch_reports_its_own_size(self):
        self.assertEqual(self._fb(5).real_batch_size, 5)

    def test_padded_batch_reports_pre_padding_size(self):
        # prepare_mlp_sync_batch grew 3 real requests to 8 padded rows.
        self.assertEqual(self._fb(8, original=3).real_batch_size, 3)

    def test_zero_real_requests_is_preserved(self):
        # An idle DP rank padded up to peer width must still report 0 real
        # requests, or every padded block would be executed.
        self.assertEqual(self._fb(8, original=0).real_batch_size, 0)


if __name__ == "__main__":
    unittest.main()
