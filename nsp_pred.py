#@title NSP Prediction
def show_nsp_predictions(model, tokenizer, sentence_pairs):
    model.eval()
    device = next(model.parameters()).device
    label_map = {0: "IsNext", 1: "NotNext"}

    for sentence_a, sentence_b in sentence_pairs:
        encoding = tokenizer(sentence_a, sentence_b, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        with torch.no_grad():
            _, nsp_logits = model(input_ids, token_type_ids)

        predicted_label = torch.argmax(nsp_logits, dim=-1).item()

        print(f"Sentence A: {sentence_a}")
        print(f"Sentence B: {sentence_b}")
        print("Predicted NSP Label:", label_map[predicted_label])
        print("-" * 50)

# Test sentences list
sentence_pairs = [
    ("The sky is blue.", "The sun is bright."),
    ("I love pizza.", "It is raining outside."),
    ("She went to the store.", "She bought some milk."),
    ("The cat is sleeping.", "Dogs bark loudly."),
    ("This is a test.", "This is not related.")
]

show_nsp_predictions(model, tokenizer, sentence_pairs)
