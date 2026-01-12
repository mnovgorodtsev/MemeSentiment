import ollama


prompts = ["Classify if this mem is funny or not funny. Return only class: funny or not funny",
           "Classify if this mem is general, twisted or not sarcastic. Return only class: general, twisted or not sarcastic",
           "Classify if this mem is offensive or not offensive. Return only class: offensive or not offensive",
           "Classify if this mem is motivational or not motivational. Return only class: motivational or "
           "not motivational"
           ]

for prompt in prompts:
    response = ollama.chat(model='qwen3-vl:2b',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': ["./data/memotion_dataset_7k/images/image_46.jpg"]
        }],
        )
    print(response['message']['content'])