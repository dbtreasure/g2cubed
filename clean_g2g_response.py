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
    terms: list[str]
# main init

toss_example = G2GTerm(
    term = "Pivot",
    snippet = "And then pivoted to the enterprise",
    definition = "A fundamental change in a company's business strategy or product direction, often in response to market feedback or challenges in the original approach. Companies may pivot to explore new markets, adopt different technologies, or alter their products to better meet customer needs.",
    explanation = "The speaker is discussing how the company changed its focus or strategy by shifting towards serving enterprise customers instead of their original target market. This change was likely in response to market conditions or opportunities that they identified."
)

keep_example = G2GTerm(
    term = "Software Multiples",
    snippet = "One thing I'd love to talk about, especially with you here, Aaron, is software multiples",
    definition = "A valuation metric used to estimate the value of a software company relative to its revenue. Software multiples are calculated by comparing the company's market value (such as market cap or enterprise value) to its revenue, often used to assess whether a company is overvalued or undervalued in comparison to its peers.",
    explanation = "The discussion turns to assessing the value of software companies by comparing their total value to their income. This is a common method used to gauge if a software company's market price is high or low compared to how much money it makes."
)

if __name__ == "__main__":
    # open g2g-response.json and read the file
    with open("g2g-keepers.json", "r", encoding="utf-8") as file:
        response = json.load(file)
        g2g_terms = [G2GTerm(**term) for term in response]

    def run_chat(terms: G2GTerm) -> None:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_model=G2GResponse,
            messages=[
                {"role": "system", "content": "You are an editorial assistant helping \
                    examine definitions generated from a podcast transcript. You are are \
                    considering a technical and software literate audience and will decide \
                    whether the supplied defifition is interesting and helps teach software \
                    professionals about investing and business management beyond just simple terms \
                    . You will be given a list of terms that have been de-duplicated and gone \
                    through a first pass of being interesting. You will respond in a JSON format \
                    with a list from most interesting to least interesting."},
                {"role": "user", "content": f"Look at the following term and definition. Look at the \
                terms and return them in a list from most interesting to least interesting. \n \
                Definition : \n {terms} \n Respond in a JSON format with a list from most interesting \
                to least interesting."},
            ]
        )

        assert isinstance(response, G2GResponse)
        return response
    
    terms = " \n ".join(f"{term.term}" for term in g2g_terms)    
    chat_response = run_chat(terms)
    
    with open("g2g-keepers-ranked.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(chat_response.terms, indent=4))
            