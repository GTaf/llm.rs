use tiktoken_rs::r50k_base;

fn main() {
    let tokenizer = r50k_base().unwrap();
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    println!("{}", tokens.len());
    println!("{}", tokens[0])
}
