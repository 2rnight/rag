"""
RAG 核心引擎 - 支持多格式文档、持久化、对话历史
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


import config


class RAGEngine:
    """RAG 问答引擎"""

    # 支持的文件格式和对应的加载器
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
        ".html": UnstructuredHTMLLoader,
    }

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.EMBEDDING_API_KEY,    # ← 硅基流动
            openai_api_base=config.EMBEDDING_BASE_URL,
        )
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.OPENAI_BASE_URL,
        )
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " "],
        )

        # 尝试加载已有的向量数据库
        self._load_existing_index()

    def _load_existing_index(self):
        """加载已有的 FAISS 索引"""
        index_path = config.FAISS_INDEX_PATH
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ 已加载向量数据库，共 {self.vectorstore.index.ntotal} 个向量")
            except Exception as e:
                print(f"⚠️ 加载向量数据库失败: {e}")

    # ==================== 文档处理 ====================

    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        ext = Path(file_path).suffix.lower()
        if ext not in self.LOADER_MAPPING:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {list(self.LOADER_MAPPING.keys())}")

        loader_class = self.LOADER_MAPPING[ext]
        
        if ext == ".txt":
            loader = loader_class(file_path, encoding="utf-8")
        else:
            loader = loader_class(file_path)

        documents = loader.load()
        
        # 添加文件名到 metadata
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(file_path)

        return documents

    def load_from_url(self, url: str) -> List[Document]:
        """从网页 URL 加载文档"""
        loader = WebBaseLoader(url)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_url"] = url
        return documents

    def load_directory(self, directory: str) -> List[Document]:
        """加载目录下所有支持的文档"""
        all_documents = []
        supported_extensions = set(self.LOADER_MAPPING.keys())

        for file_path in Path(directory).rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_document(str(file_path))
                    all_documents.extend(docs)
                    print(f"  ✅ 已加载: {file_path.name} ({len(docs)} 页)")
                except Exception as e:
                    print(f"  ❌ 加载失败: {file_path.name} - {e}")

        return all_documents

    # ==================== 索引构建 ====================

    def build_index(self, documents: List[Document]) -> int:
        """构建向量索引"""
        if not documents:
            raise ValueError("没有文档可以索引")

        # 分割文档
        chunks = self.text_splitter.split_documents(documents)
        print(f"📄 文档分割为 {len(chunks)} 个片段")

        # 构建向量数据库
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            # 增量添加
            self.vectorstore.add_documents(chunks)

        # 持久化保存
        self.vectorstore.save_local(config.FAISS_INDEX_PATH)
        print(f"💾 向量数据库已保存，共 {self.vectorstore.index.ntotal} 个向量")

        return len(chunks)

    def build_from_directory(self, directory: str = None) -> int:
        """从目录构建索引"""
        directory = directory or config.DOCS_DIR
        print(f"📂 正在加载目录: {directory}")
        documents = self.load_directory(directory)
        print(f"📄 共加载 {len(documents)} 个文档页面")
        return self.build_index(documents)

    def add_file(self, file_path: str) -> int:
        """增量添加单个文件"""
        documents = self.load_document(file_path)
        return self.build_index(documents)

    def add_url(self, url: str) -> int:
        """增量添加网页"""
        documents = self.load_from_url(url)
        return self.build_index(documents)

    def reset_index(self):
        """重置向量数据库"""
        self.vectorstore = None
        import shutil
        if os.path.exists(config.FAISS_INDEX_PATH):
            shutil.rmtree(config.FAISS_INDEX_PATH)
        print("🗑️ 向量数据库已重置")

    # ==================== 问答 ====================

    def query(self, question: str, chat_history: List[Dict] = None) -> Dict:
        """
        问答查询
        
        Args:
            question: 用户问题
            chat_history: 对话历史 [{"role": "user", "content": "..."}, ...]
        
        Returns:
            {"answer": str, "sources": List[Dict]}
        """
        if self.vectorstore is None:
            return {
                "answer": "❌ 知识库为空，请先上传文档。",
                "sources": []
            }

        # 检索相关文档
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_K}
        )

        # 构建提示模板
        system_template = """你是一个智能文档问答助手。根据提供的上下文信息回答用户的问题。

规则：
1. 只根据提供的上下文信息回答，不要编造内容
2. 如果上下文中没有相关信息，请明确说明"根据现有文档，未找到相关信息"
3. 回答要准确、简洁、有条理
4. 如果涉及具体数据或规定，请引用原文
5. 使用中文回答

上下文信息：
{context}
"""

        if chat_history:
            history_text = "\n".join([
                f"{'用户' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
                for msg in chat_history[-6:]  # 只保留最近6轮
            ])
            system_template += f"\n\n之前的对话记录：\n{history_text}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])

        # 构建链
        def format_docs(docs):
            return "\n\n---\n\n".join([
                f"[来源: {doc.metadata.get('source_file', '未知')} | "
                f"页码: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            ])

        # 先检索文档
        retrieved_docs = retriever.invoke(question)

        chain = (
            {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        # 整理来源信息
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "file": doc.metadata.get("source_file", doc.metadata.get("source", "未知")),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:200] + "...",
                "score": None,  # FAISS 默认不返回分数
            })

        return {
            "answer": answer,
            "sources": sources
        }

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """纯相似度搜索（不经过 LLM）"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        if self.vectorstore is None:
            return {"total_vectors": 0, "status": "empty"}
        return {
            "total_vectors": self.vectorstore.index.ntotal,
            "status": "ready"
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    engine = RAGEngine()

    # 构建索引
    engine.build_from_directory("./docs")

    # 交互式问答
    print("\n" + "=" * 50)
    print("📚 智能文档问答系统已就绪！输入 'quit' 退出")
    print("=" * 50)

    chat_history = []

    while True:
        question = input("\n🙋 你的问题: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue

        result = engine.query(question, chat_history)

        print(f"\n🤖 回答: {result['answer']}")
        print(f"\n📎 参考来源 ({len(result['sources'])} 条):")
        for i, src in enumerate(result["sources"], 1):
            print(f"   {i}. {src['file']} (第{src['page']}页)")

        # 记录对话历史
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result["answer"]})