%%time
import textwrap
response = ''
instruction = 'who is vishal mysore.'
generated_text = pipe(instruction)
for text in generated_text:
  response += text['generated_text']
wrapped_text = textwrap.fill(response, width=80)
print(wrapped_text)