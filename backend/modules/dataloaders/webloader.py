import os
import shutil
from typing import List

import markdownify
from bs4 import BeautifulSoup as Soup
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

from backend.modules.dataloaders.loader import BaseLoader
from backend.utils.base import DocumentMetadata, LoadedDocument, SourceConfig
from backend.utils.utils import generate_document_id
from backend.utils.logger import logger


class WebLoader(BaseLoader):
    """
    This loader handles web URLs
    """

    type = "web"

    def _remove_empty_lines(self, text):
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip() != ""]
        return "\n".join(non_empty_lines)

    def _remove_tags(self, html):
        # parse html content
        soup = Soup(html, "html.parser")

        for data in soup(["style", "script"]):
            # Remove tags
            data.decompose()

        # return data by retrieving the tag content
        return " ".join([str(d) for d in soup])

    def load_data(
        self, source_config: SourceConfig, dest_dir: str, allowed_extensions: List[str]
    ) -> List[LoadedDocument]:
        """
        Loads data from a local directory specified by the given source URI.

        Args:
            source_config (SourceConfig): The source URI of the website
            dest_dir (str): The destination directory where the data will be copied to.
            allowed_extensions (List[str]): A list of allowed file extensions.
        Returns:
            List[LoadedDocument]: A list of LoadedDocument objects containing metadata.
        """
        max_depth = source_config.dict().get("max_depth", 2)
        logger.info("WebLoader -> max_depth: ", max_depth)
        loader = RecursiveUrlLoader(
            url=source_config.uri,
            max_depth=max_depth,
            extractor=lambda x: self._remove_empty_lines(
                markdownify.markdownify(self._remove_tags(x))
            ),
        )
        documents = loader.load()

        loaded_documents: List[LoadedDocument] = []
        for i, doc in enumerate(documents):
            title: str = doc.metadata.get("title")
            url: str = doc.metadata.get("source")
            if title is None or url is None:
                continue

            file_name = f"file_{i}.md"
            dest_path = os.path.join(dest_dir, file_name)

            with open(dest_path, "w") as f:
                f.write(doc.page_content)

            document_id = generate_document_id(self.type, source_config.uri, url)

            loaded_documents.append(
                LoadedDocument(
                    filepath=dest_path,
                    file_extension=".md",
                    metadata=DocumentMetadata(
                        document_id=document_id,
                        title=title,
                        source=url,
                        description=doc.metadata.get("description"),
                    ),
                )
            )

        return loaded_documents
