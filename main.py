from detector import predict_message, train_model


def main() -> None:
    artifacts = train_model()
    dataset = artifacts["dataset"]

    print("Dataset size:", dataset.shape)
    print(dataset["label"].value_counts().rename({0: "ham", 1: "spam"}))
    print(f"\nModel Accuracy: {artifacts['accuracy']:.4f}")
    print("\nEnter a message to check:")

    user_input = input().strip()
    result = predict_message(
        user_input,
        artifacts["model"],
        artifacts["vectorizer"],
    )

    if result["is_scam"]:
        print("Possible Scam Message")
    else:
        print("Message looks Safe")

    print(f"Scam Probability: {result['scam_probability']:.2%}")

    if result["keyword_hits"]:
        print("Suspicious keywords:", ", ".join(result["keyword_hits"]))


if __name__ == "__main__":
    main()
