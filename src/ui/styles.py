APP_STYLES = """
<style>

section[data-testid="stSidebar"] {
    width: 320px !important;
}

div.stButton > button {
    width: 100%;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-radius: 10px;
    margin-bottom: 6px;
    padding: 0.5rem;
}

.user-message {
    background-color: #3A3A3C;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 12px;
    color: #FFFFFF;
}

.assistant-message {
    background-color: #1C1C1E;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid #2C2C2E;
    color: #F2F2F7;
}

.status-message {
    background-color: #1E1E1E;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid #333333;
    opacity: 0.7;
    font-style: italic;
}

.source-preview {
    color: #D1D5DB;
    font-size: 14px;
    line-height: 1.5;
}

.source-card {
    background: linear-gradient(
        145deg,
        #0F172A,
        #111827
    );
    border: 1px solid #243041;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}

.source-title {
    font-weight: 700;
    font-size: 16px;
    color: #F9FAFB;
    margin-bottom: 14px;
}

.source-preview-box {
    background-color: #1E293B;
    border-radius: 12px;
    padding: 14px;
    color: #D1D5DB;
    font-size: 14px;
    line-height: 1.7;
    border-left: 4px solid #3B82F6;
    overflow-wrap: break-word;
}

</style>
"""