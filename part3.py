# Python

from openai import OpenAI
from pinecone import Pinecone

class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = "Evaluate if the user's query is obnoxious or not. An obnoxious query is one that is rude, offensive, or inappropriate. Return True if the query is obnoxious, and False otherwise."

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        if "True" in response:
            return True
        return False

    def check_query(self, query):
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
        )
        return self.extract_action(response.choices[0].message.content)


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        self.client = openai_client

    def rephrase(self, user_history, latest_query):
        prompt = f"History: {user_history}\nLatest Query: {latest_query}\nRephrase the latest query to be self-contained and clear without losing its original meaning."
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.prompt = "Retrieve the most relevant segments for this query"

    def query_vector_store(self, query, k=5):
        res = self.client.embeddings.create(input=[query], model="text-embedding-3-small")
        xq = res.data[0].embedding
        # Search Pinecone
        res = self.index.query(vector=xq, top_k=k, include_metadata=True, namespace= "ns2500")
        return [match['metadata']['text'] for match in res['matches']]

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response, query = None):
        return response


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def generate_response(self, query, docs, conv_history, k=5):
        context = "\n".join(docs[:k])
        system_prompt = f"""You are a friendly and helpful Machine Learning chatbot.

If the user's query is a greeting or small talk (e.g., 'Hi', 'How are you?', 'sup', 'thanks'), respond warmly and naturally without using the context below. Keep it brief and friendly.

For technical or specific questions about Machine Learning, use the following context to provide accurate answers:
{context}"""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conv_history)
        messages.append({"role": "user", "content": query})
        
        response = self.client.chat.completions.create(model="gpt-4.1-nano", messages=messages)
        return response.choices[0].message.content

class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def get_relevance(self, query, documents) -> str:
        context = "\n".join(documents)
        prompt = f"User Query: {query}\n\nRetrieved Documents:\n{context}\n\nFirst, check if this is a greeting or small talk (e.g., 'Hi', 'Hello', 'How are you?', 'Thanks', 'sup'). If it is, respond with ONLY the word: SMALL_TALK\n\nOtherwise, for technical queries, return 'Yes' if the documents are relevant, or 'No' if they are not relevant. Your only return opitons are Yes, No, or SMALL_TALK."
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        self.client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(pinecone_index_name)
        self.setup_sub_agents()
        self.history = []
        self.latest_user_query = None

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_rewriter_agent = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.client, None)  
        self.answering_agent = Answering_Agent(self.client)
        self.relevant_docs_agent = Relevant_Documents_Agent(self.client)


    def main_loop(self):
        if not self.latest_user_query:
            return "No query provided."
        if self.obnoxious_agent.check_query(self.latest_user_query):
            return "Your query was flagged as obnoxious. Please rephrase and try again."
        
        # Rephrase query with context from history to handle pronouns
        rephrased_query = self.context_rewriter_agent.rephrase(
            self.history,
            self.latest_user_query
        )
        
        search_results = self.query_agent.query_vector_store(rephrased_query)
        relevance = self.relevant_docs_agent.get_relevance(rephrased_query, search_results)
        
        # Handle small talk separately - no search results or history needed
        if relevance == "SMALL_TALK":
            response = self.answering_agent.generate_response(
                self.latest_user_query, 
                [],  # empty search results
                []   # empty history
            )
            self.history.append({"role": "user", "content": self.latest_user_query})
            self.history.append({"role": "assistant", "content": response})
            return response
        
        if relevance.lower() == "no":
            return "I'm sorry, I don't have relevant information to answer your query at the moment."
        
        response = self.answering_agent.generate_response(rephrased_query, search_results, self.history)
        self.history.append({"role": "user", "content": self.latest_user_query})
        self.history.append({"role": "assistant", "content": response})
        return response