#@title MLM Prediction
def show_mlm_predictions(model, tokenizer, sentences):
    model.eval()
    device = next(model.parameters()).device

    for sentence in sentences:
        encoding = tokenizer(sentence, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        with torch.no_grad():
            mlm_logits, _ = model(input_ids, token_type_ids)

        mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = mlm_logits[0, mask_token_index].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        reconstructed = original_tokens.copy()
        reconstructed[mask_token_index.item()] = predicted_token

        print("Original Sentence: ", sentence)
        print("Predicted Token: ", predicted_token)
        print("Full Reconstructed Sentence: ", tokenizer.convert_tokens_to_string(reconstructed))
        print("-" * 50)

# Test sentences list
sentences = [
    "The capital of France is [MASK].",
    "The president of the United States is [MASK].",
    "Water freezes at [MASK] degrees Celsius.",
    "Python is a popular [MASK] programming language.",
    "The sky is [MASK]."
]

model = results["model"]
show_mlm_predictions(model, tokenizer, sentences)
