from helpers.gpt import client

print(client.batches.retrieve(input("Enter batch ID: ")).status)
