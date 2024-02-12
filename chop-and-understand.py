from openai import OpenAI
import json
import instructor
from pydantic import BaseModel

openai_client = instructor.patch(OpenAI())

class G2GTerm(BaseModel):
    term: str
    snippet: str
    definition: str
    explanation: str

class G2GResponse(BaseModel):
    terms: list[G2GTerm]

# main init 
if __name__ == "__main__":

    example_snippet = "Most people in Silicon Valley know you well. But maybe just level set, you know, quick reminder for everybody. What does Box do? What inspired you to get it started and then how it's evolving, you know, as you guys exit Zerp. Yeah. Actually, the Zerp thing. I mean, I know there's a lot of Zerp stuff going on. But we actually never really even had that much of a Zerp benefit."
    example_response = """
    [
        {
            "term": "Free Cash Flow",
            "snippet": "On a free cash flow basis, you know, the multiples are actually trendy right around that 10-year average.",
            "definition": "The amount of cash a company generates after accounting for cash outflows to support operations and maintain its capital assets. It's a measure of financial performance that shows how much cash is available for the company to repay creditors or pay dividends and interest to investors.",
            "explanation": "The discussion points out that when looking at companies from the perspective of how much usable cash they generate (after all expenses and investments), the valuation multiples (how much the company is worth compared to its free cash flow) align closely with the averages seen over the past decade. This means that despite various economic changes, the value investors place on companies, based on the cash they can freely use, remains consistent with long-term trends. It suggests a stable investor sentiment towards how businesses manage and generate cash beyond their immediate operational and investment needs."
        },
        {
            "term": "SaaS (Software as a Service)",
            "snippet": "I think the average multiple for a SaaS company is now six.",
            "definition": "A software distribution model where applications are hosted by a vendor and made available to customers over the internet, typically on a subscription basis.",
            "explanation": "In the context of the podcast, when they mention 'the average multiple for a SaaS company is now six,' they're discussing how the market values these companies compared to their revenue. Essentially, a SaaS company's market value is estimated to be six times its annual revenue, illustrating investor valuation of these companies based on their earnings."
        }
    ]
    """
    example_no_response = "I can absolutely see sometime over the next four quarters. We're going to hit a zone of dissolution, because everybody's pig piled in here. They think it's all happening now. It's going to take a little bit longer just like the internet did, just like cloud did. We made it to episode two. We did. Aaron, I'm super excited. You decided to join us. I've known you for a while and I've heard other people talk about you and there's two things I just really love. One, you always have an independent point of view and you're not afraid to state it, which is not true for everyone, especially people in the CEO slot."
    chunks = []
    terms = []
    responses = []
    SENTENCES_PER_CHUNK = 16
    OVERLAP = 2

    def split_text() -> None:
        with open("transcription-2.txt", "r", encoding="utf-8") as file:
            text = file.read()
            sentences = text.split(".")
            for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
                chunk = " ".join(sentences[i:i+SENTENCES_PER_CHUNK+OVERLAP])
                chunks.append(chunk)

    def run_chat(snippet: str, terms: list) -> None:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_model=G2GResponse,
            messages=[
                {"role": "system", "content": f"You are an analyst helping me to understand a transcript of \
                    a podcast. There are terms and concepts that relate to economics and finance that I \
                    need a better understanding of. I'll be supplying you with snippets and I am asking \
                    you to identify economic and financial terms and if you find any you will return \
                    them in a JSON array with each term a JSON object with a definition, the precise section of the snippet it came from, and a \
                    re-phrasing of the snippet in simpler terms in JSON format. \n Here is an example: \n # Snippet : \n \
                    {example_snippet} \n (Response) \n {example_response} \n Here is an example with no \
                    economic or financial terms: \n # Snippet : \n {example_no_response} \n (Response) {{[]}} \
                    I will also supply you with terms that have already been covered so don't repeat them \
                    if they are basically the same. \n So to review return {{[]}} if there are no \
                    economic or financial terms in the snippet. "},
                {"role": "user", "content": f"Look at the following section of the podcast. There may \
                    or may not be economic and financial terms in the snippet. If you find any, please \
                    follow instructions. Here are the terms you don't need to cover: \n Already covered \
                    terms: \n {terms} \n Now here is the snippet. Remember to provide the same format \
                    for each term with a `Term`, `Snippet`, `Definition` and `Simplified Explanation of What They Are Saying` \
                    \n # Snippet : \n {snippet}"},
            ]
        )

        assert isinstance(response, G2GResponse)
        for term in response.terms:
            terms.append(term.term)
            responses.append(term)
            with open("g2g-response-test.json", "w", encoding="utf-8") as file:
                term_dicts = [term.model_dump() for term in responses]
                file.write(json.dumps(term_dicts, indent=4))
    
    split_text()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i +1} of {len(chunks)}")
        retry_count = 0
        try:
            run_chat(chunk, terms)

        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            print("retrying...")
            if retry_count < 3:
                retry_count += 1
                run_chat(chunk, terms)
                continue
    
    with open("g2g-response.json", "w", encoding="utf-8") as file:
        term_dicts = [term.dict() for term in responses]
        file.write(json.dumps(term_dicts, indent=4))
