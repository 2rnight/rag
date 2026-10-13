"""
Streamlit Web 界面
运行: streamlit run app.py
"""
import streamlit as st
import tempfile
import os
from rag_engine import RAGEngine

# ===== 页面配置 =====
st.set_page_config(
    page_title="📚 智能文档问答系统",
    page_icon="📚",
    layout="wide"
)

# ===== 初始化 =====
@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# 初始化会话状态
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== 侧边栏 =====
with st.sidebar:
    st.title("⚙️ 设置")

    # 知识库状态
    stats = engine.get_stats()
    st.metric("知识库向量数", stats["total_vectors"])
    st.caption(f"状态: {'✅ 就绪' if stats['status'] == 'ready' else '⚠️ 空'}")

    st.divider()

    # 文件上传
    st.subheader("📁 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF、Word、TXT、CSV",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🔨 处理文档", type="primary"):
        with st.spinner("正在处理文档..."):
            for uploaded_file in uploaded_files:
                # 保存到临时文件
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    num_chunks = engine.add_file(tmp_path)
                    st.success(f"✅ {uploaded_file.name}: {num_chunks} 个片段")
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

        st.rerun()

    st.divider()

    # URL 添加
    st.subheader("🌐 添加网页")
    url = st.text_input("输入网页 URL")
    if url and st.button("🔗 添加网页"):
        with st.spinner("正在抓取网页..."):
            try:
                num_chunks = engine.add_url(url)
                st.success(f"✅ 已添加 {num_chunks} 个片段")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 添加失败: {e}")

    st.divider()

    # 批量加载目录
    if st.button("📂 加载 docs 目录"):
        with st.spinner("正在加载..."):
            try:
                num = engine.build_from_directory("./docs")
                st.success(f"✅ 已加载 {num} 个片段")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    # 重置
    if st.button("🗑️ 重置知识库", type="secondary"):
        engine.reset_index()
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

# ===== 主界面 =====
st.title("📚 智能文档问答系统")
st.caption("基于 RAG (Retrieval-Augmented Generation) 技术")

# 显示对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 参考来源"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src['file']} | 第 {src['page']} 页")
                    st.text(src["content"])

# 用户输入
if question := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = engine.query(question, st.session_state.chat_history)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📎 参考来源"):
                for src in result["sources"]:
                    st.caption(f"📄 {src['file']} | 第 {src['page']} 页")
                    st.text(src["content"])

    # 保存到历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})