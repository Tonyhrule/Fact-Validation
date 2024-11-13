from helpers.oai import client

print(client.batches.retrieve(input("Enter batch ID: ")).status)
