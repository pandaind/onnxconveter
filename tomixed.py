from onnxconverter_common import auto_mixed_precision
import onnx

model = onnx.load("codeformer.onnx")
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model,feed_dict)
#model_fp16 = auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
#auto_convert_mixed_precision(model, feed_dict, validate_fn=None, rtol=None, atol=None, keep_io_types=False)
onnx.save(model_fp16, "codeformer-f16.onnx")
