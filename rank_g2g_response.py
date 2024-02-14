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
    keep: bool
# main init

if __name__ == "__main__":
    # open g2g-response.json and read the file
    with open("g2g-response.json", "r", encoding="utf-8") as file:
        response = json.load(file)
        g2g_terms = [G2GTerm(**term) for term in response]

    keepers = []
    def run_chat(term: G2GTerm) -> None:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            response_model=G2GResponse,
            messages=[
                {"role": "system", "content": f"You are an editorial assistant helping \
                    examine definitions generated from a podcast transcript. You are are \
                    considering a technical and software literate audience and will decide \
                    whether the supplied defifition is interesting and helps teach software \
                    professionals about investing and business management beyond just simple terms \
                    . You will respond in a JSON format with a boolean value under a key of `keep`. \
                    Your criteria for keeping is that the definition is not a duplicate from the list \
                    of what has been kept so far that you'll be provided with. This can be an exact duplicate \
                    or something that has roughly already been covered. Air on the side of being critical. \
                    You are trying to end up with novel terms that aren't quite household. \
                    \n Here is an example: \n # Definition : \n  {dict(toss_example)} \n \
                    Response: \n {{keep: false }} \n This example is a little broad and shoudl be tossed.\
                    Other terms you'd toss are 'business model', or things that aren't definitions like \
                    'H1B demand'. \
                    \n Here is an example of a good definition: \n # Definition : \n {dict(keep_example)} \
                    \n Response: \n {{keep: true }} \n This example is specific and should be kept."},
                {"role": "user", "content": f"Look at the following term and definition. Is it something \
                    that's engaging, a little beyond the basics, and would be interesting to a technical audience? \
                    \n Here are the terms so far: \n {[term.term for term in keepers]} \n \
                    \n # Definition : \n {dict(term)}"},
            ]
        )

        assert isinstance(response, G2GResponse)

        if response.keep:
            print(f"Keeping {term.term}!")
            keepers.append(term)
        else:
            print(f"Tossing {term.term}!")
    
    for term in g2g_terms:
        run_chat(term)
    
    with open("g2g-keepers.json", "w", encoding="utf-8") as file:
        term_dicts = [term.dict() for term in keepers]
        file.write(json.dumps(term_dicts, indent=4))
            