import argparse

from tf_require import require_tensorflow

tf = require_tensorflow()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras_model", required=True, help="Path to a .keras model (prefer embedding model)")
    ap.add_argument("--out", required=True, help="Output .tflite path")
    ap.add_argument("--quant", choices=["none", "float16"], default="float16")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.keras_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.quant == "float16":
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(args.out, "wb") as f:
        f.write(tflite_model)

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
