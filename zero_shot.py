from openai import OpenAI


API_SECRET_KEY = "sk-zk23f6f0433fac7113e038a12d7480b7f466826b78ae8c15"
BASE_URL = "https://api.zhizengzeng.com/v1"
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

# files
def create_files():
    resp = client.files.create(
        file=open("./data/MAVEN/requests.jsonl", "rb"),
        purpose='batch'
    )
    print(resp)
    return resp.id


def create_batches(file_id):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(resp)
    return resp.id


if __name__ == "__main__":
    # file_id = create_files()
    # print(file_id)
    # file_id = 'file-6826b985bd1bc17473683255'
    # batch_id = create_batches(file_id)
    # print(batch_id)
    batch_id = 'batch_6826b9a04471017473683520'
    resp = client.batches.retrieve(batch_id)
    print(resp)
    # resp = client.files.content(file_id)
    # print(resp)