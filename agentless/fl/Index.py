import copy
import os
from abc import ABC

from llama_index.core import (
    Document,
    Settings,
)
from llama_index.core.schema import MetadataMode
from dotenv import load_dotenv

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.index_skeleton import parse_global_stmt_from_code
from agentless.util.preprocess_data import (
    clean_method_left_space,
    get_full_file_paths_and_classes_and_functions,
)
from get_repo_structure.get_repo_structure import parse_python_file
from google import genai
import numpy as np
import faiss


def construct_file_meta_data(file_name: str, clazzes: list, functions: list) -> dict:
    meta_data = {
        "file_name": file_name,
    }
    meta_data["File Name"] = file_name

    if clazzes:
        meta_data["Classes"] = ", ".join([c["name"] for c in clazzes])
    if functions:
        meta_data["Functions"] = ", ".join([f["name"] for f in functions])

    return meta_data


def check_meta_data(meta_data: dict) -> bool:

    doc = Document(
        text="",
        metadata=meta_data,
        metadata_template="### {key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
    )

    if (
        num_tokens_from_messages(
            doc.get_content(metadata_mode=MetadataMode.EMBED),
            model="text-embedding-3-small",
        )
        > Settings.chunk_size // 2
    ):
        # half of the chunk size should not be metadata
        return False

    return True


def build_file_documents_simple(
    clazzes: list, functions: list, file_name: str, file_content: str
) -> list[Document]:
    """
    Really simple file document format, where we put all content of a single file into a single document
    """
    documents = []

    meta_data = construct_file_meta_data(file_name, clazzes, functions)

    doc = Document(
        text=file_content,
        metadata=meta_data,
        metadata_template="### {key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
    )
    doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
    doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
    if not check_meta_data(meta_data):
        # meta_data a bit too long, instead we just exclude meta data
        doc.excluded_embed_metadata_keys = list(meta_data.keys())
        doc.excluded_llm_metadata_keys = list(meta_data.keys())
        documents.append(doc)
    else:
        documents.append(doc)

    return documents


def build_file_documents_complex(
    clazzes: list, functions: list, file_name: str, file_content: str
) -> list[Document]:

    documents = []

    global_stmt, _ = parse_global_stmt_from_code(file_content)
    base_meta_data = construct_file_meta_data(file_name, clazzes, functions)

    for clazz in clazzes:
        content = "\n".join(clazz["text"])
        meta_data = copy.deepcopy(base_meta_data)
        meta_data["Class Name"] = clazz["name"]
        doc = Document(
            text=content,
            metadata=meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )

        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(meta_data):
            doc.excluded_embed_metadata_keys = list(meta_data.keys())
            doc.excluded_llm_metadata_keys = list(meta_data.keys())
        documents.append(doc)

        for class_method in clazz["methods"]:
            method_meta_data = copy.deepcopy(base_meta_data)
            method_meta_data["Class Name"] = clazz["name"]
            method_meta_data["Method Name"] = class_method["name"]
            content = clean_method_left_space("\n".join(class_method["text"]))

            doc = Document(
                text=content,
                metadata=method_meta_data,
                metadata_template="### {key}: {value}",
                text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
            )
            doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
            doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
            if not check_meta_data(method_meta_data):
                doc.excluded_embed_metadata_keys = list(method_meta_data.keys())
                doc.excluded_llm_metadata_keys = list(method_meta_data.keys())
            documents.append(doc)

    for function in functions:
        content = "\n".join(function["text"])
        function_meta_data = copy.deepcopy(base_meta_data)
        function_meta_data["Function Name"] = function["name"]
        doc = Document(
            text=content,
            metadata=function_meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )

        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(function_meta_data):
            doc.excluded_embed_metadata_keys = list(function_meta_data.keys())
            doc.excluded_llm_metadata_keys = list(function_meta_data.keys())
        documents.append(doc)

    if global_stmt != "":
        content = global_stmt
        global_meta_data = copy.deepcopy(base_meta_data)

        doc = Document(
            text=content,
            metadata=global_meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )
        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(global_meta_data):
            doc.excluded_embed_metadata_keys = list(global_meta_data.keys())
            doc.excluded_llm_metadata_keys = list(global_meta_data.keys())
        documents.append(doc)

    return documents


class EmbeddingIndex(ABC):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        filter_type,
        index_type,
        chunk_size,
        chunk_overlap,
        logger,
        **kwargs,
    ):
        self.instance_id = instance_id
        self.structure = structure
        self.problem_statement = problem_statement
        self.filter_type = filter_type
        self.index_type = index_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        self.kwargs = kwargs

    def filter_files(self, files):
        if self.filter_type == "given_files":
            given_files = self.kwargs["given_files"][: self.kwargs["filter_top_n"]]
            return given_files
        elif self.filter_type == "none":
            # all files are included
            return [file_content[0] for file_content in files]
        else:
            raise NotImplementedError

    def chunk_documents(self, documents):
        all_chunks = []
        doc_ids = []

        for doc_id, doc in enumerate(documents):
            chunks = []
            # TODO: not sure if original does chunk_size tokens or characters...
            for i in range(0, len(doc.text), self.chunk_size - self.chunk_overlap):
                chunk = doc.text[i : i + self.chunk_size]
                chunks.append(chunk)

            all_chunks.extend(chunks)
            doc_ids.extend([doc_id] * len(chunks))

        return all_chunks, doc_ids

    def retrieve(self, mock=False):
        files, _, _ = get_full_file_paths_and_classes_and_functions(self.structure)
        filtered_files = self.filter_files(files)
        self.logger.info(f"Total number of considered files: {len(filtered_files)}")
        print(f"Total number of considered files: {len(filtered_files)}")
        documents = []
        for file_content in files:
            content = "\n".join(file_content[1])
            file_name = file_content[0]

            if file_name not in filtered_files:
                continue

            class_info, function_names, _ = parse_python_file(None, content)
            if self.index_type == "simple":
                docs = build_file_documents_simple(
                    class_info, function_names, file_name, content
                )
            elif self.index_type == "complex":
                docs = build_file_documents_complex(
                    class_info, function_names, file_name, content
                )
            else:
                raise NotImplementedError

            documents.extend(docs)

        self.logger.info(f"Total number of documents: {len(documents)}")
        print(f"Total number of documents: {len(documents)}")
        if len(documents) == 0:
            return [], []

        load_dotenv()
        client = genai.Client(api_key=os.getenv("API_KEY"))

        chunks, doc_ids = self.chunk_documents(documents)

        embeddings = []
        # this api excepts batches of max 100 chunks
        for i in range(0, len(chunks), 100):
            embeddings.extend(
                client.models.embed_content(
                    model="text-embedding-004", contents=chunks[i : i + 100]
                ).embeddings
            )

        embeddings_np = np.array([e.values for e in embeddings], dtype=np.float32)
        dimension = embeddings_np.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)

        query_embedding = (
            client.models.embed_content(
                model="text-embedding-004", contents=self.problem_statement
            )
            .embeddings[0]
            .values
        )
        query_vec = np.array([query_embedding], dtype=np.float32)

        distances, indices = index.search(query_vec, k=100)

        chunks = [chunks[i] for i in indices[0]]
        doc_ids = [doc_ids[i] for i in indices[0]]

        file_names = []
        meta_infos = []
        for chunk, doc_id in zip(chunks, doc_ids):
            file_name = documents[doc_id].metadata["File Name"]
            if file_name not in file_names:
                file_names.append(file_name)
                self.logger.info("================")
                self.logger.info(file_name)

                self.logger.info(documents[doc_id].text)

                meta_infos.append(
                    {
                        "code": documents[doc_id].text,
                        "metadata": documents[doc_id].metadata,
                    }
                )
        return file_names, meta_infos
