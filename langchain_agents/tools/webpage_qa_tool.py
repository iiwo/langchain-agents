from langchain.docstore.document import Document
import asyncio

from langchain.agents import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.base import BaseTool

from pydantic import Field
from langchain.chains.qa_with_sources.loading import (
    BaseCombineDocumentsChain,
)


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a tiny chunk size, just to show.
        chunk_size=700,
        chunk_overlap=30,
        length_function=len,
    )


class WebpageQATool(BaseTool):
    NUM_RESULTS = 6

    name = "query_webpage"
    description = (
        "Browse a webpage and retrieve the information relevant to the question. Provide url and question to use it."
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), self.NUM_RESULTS):
            input_docs = web_docs[i : i + self.NUM_RESULTS]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError