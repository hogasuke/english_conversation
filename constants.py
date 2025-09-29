APP_NAME = "生成AI英会話アプリ"
MODE_1 = "日常英会話"
MODE_2 = "シャドーイング"
MODE_3 = "ディクテーション"
USER_ICON_PATH = "images/user_icon.jpg"
AI_ICON_PATH = "images/ai_icon.jpg"
AUDIO_INPUT_DIR = "audio/input"
AUDIO_OUTPUT_DIR = "audio/output"
PLAY_SPEED_OPTION = [2.0, 1.5, 1.2, 1.0, 0.8, 0.6]
ENGLISH_LEVEL_OPTION = ["初級者", "中級者", "上級者"]

# 学習進捗追跡用の定数
LEARNING_PROGRESS_KEYS = {
    "total_attempts": 0,  # 総回答数
    "correct_responses": 0,  # 正解数
    "accuracy_history": [],  # 正答率履歴（直近10回）
    "vocabulary_level": 1,  # 語彙レベル（1-10）
    "grammar_level": 1,  # 文法レベル（1-10）
    "fluency_level": 1,  # 流暢さレベル（1-10）
    "consecutive_success": 0,  # 連続成功回数
    "level_up_threshold": 3,  # レベルアップに必要な連続成功回数
}

# 難易度レベル定義
DIFFICULTY_LEVELS = {
    1: {"vocabulary": "basic", "grammar": "simple_present", "sentence_length": "8-12"},
    2: {"vocabulary": "basic", "grammar": "simple_past", "sentence_length": "10-14"},
    3: {"vocabulary": "intermediate", "grammar": "present_perfect", "sentence_length": "12-16"},
    4: {"vocabulary": "intermediate", "grammar": "conditional", "sentence_length": "14-18"},
    5: {"vocabulary": "intermediate", "grammar": "passive_voice", "sentence_length": "15-20"},
    6: {"vocabulary": "advanced", "grammar": "complex_tenses", "sentence_length": "16-22"},
    7: {"vocabulary": "advanced", "grammar": "subjunctive", "sentence_length": "18-24"},
    8: {"vocabulary": "advanced", "grammar": "advanced_structures", "sentence_length": "20-26"},
    9: {"vocabulary": "expert", "grammar": "idiomatic", "sentence_length": "22-28"},
    10: {"vocabulary": "expert", "grammar": "native_level", "sentence_length": "24-30"}
}

# 英語講師として自由な会話をさせ、文法間違いをさりげなく訂正させるプロンプト
SYSTEM_TEMPLATE_BASIC_CONVERSATION = """
    You are a conversational English tutor. Engage in a natural and free-flowing conversation with the user. If the user makes a grammatical error, subtly correct it within the flow of the conversation to maintain a smooth interaction. Optionally, provide an explanation or clarification after the conversation ends.
"""

# 約15語のシンプルな英文生成を指示するプロンプト（初回用）
SYSTEM_TEMPLATE_CREATE_PROBLEM = """
    Generate 1 sentence that reflect natural English used in daily conversations, workplace, and social settings:
    - Casual conversational expressions
    - Polite business language
    - Friendly phrases used among friends
    - Sentences with situational nuances and emotions
    - Expressions reflecting cultural and regional contexts

    Limit your response to an English sentence of approximately 15 words with clear and understandable context.
"""

# 過去の会話を踏まえた問題文生成を指示するプロンプト（2回目以降用）
SYSTEM_TEMPLATE_CREATE_CONTEXTUAL_PROBLEM = """
    You are an English conversation tutor creating follow-up sentences for practice exercises.
    
    Based on the conversation history, generate a natural follow-up sentence that:
    1. Connects logically to the previous conversation topics
    2. Maintains conversational flow and context
    3. Introduces new vocabulary or grammar patterns gradually
    4. Reflects natural English used in daily conversations, workplace, and social settings
    5. Considers the user's demonstrated English level from their previous responses
    
    Guidelines:
    - If the user struggled with certain grammar or vocabulary, introduce similar but slightly different patterns
    - If the user performed well, gradually increase complexity
    - Keep the sentence approximately 15 words
    - Ensure the sentence feels like a natural continuation of the conversation
    - Include situational context and emotions when appropriate
    - Adjust difficulty based on the user's English level: {english_level}
      * 初級者: Use simple present/past tense, basic vocabulary, common phrases
      * 中級者: Include perfect tenses, conditional forms, intermediate vocabulary
      * 上級者: Use complex grammar, advanced vocabulary, idiomatic expressions
    
    Generate only the English sentence without explanations.
"""

# 問題文と回答を比較し、評価結果の生成を指示するプロンプト（改良版）
SYSTEM_TEMPLATE_EVALUATION = """
    あなたは英語学習の専門家です。
    以下の「LLMによる問題文」と「ユーザーによる回答文」を比較し、詳細に分析してください：

    【LLMによる問題文】
    問題文：{llm_text}

    【ユーザーによる回答文】
    回答文：{user_text}

    【分析項目】
    1. 単語の正確性（誤った単語、抜け落ちた単語、追加された単語）
    2. 文法的な正確性
    3. 文の完成度
    4. 発音しやすさ（音韻的な類似性）

    評価は以下の形式で**必ず**出力してください：

    SCORE_ACCURACY:{accuracy_score}
    SCORE_GRAMMAR:{grammar_score}  
    SCORE_COMPLETENESS:{completeness_score}
    SCORE_OVERALL:{overall_score}

    【評価】
    ✓ 正確に再現できた部分 # 項目を複数記載
    △ 改善が必要な部分 # 項目を複数記載
    
    【アドバイス】
    次回の練習のためのポイント

    注意：スコアは0-100の整数で評価し、SCORE_で始まる行は必須です。
    ユーザーの努力を認め、前向きな姿勢で次の練習に取り組めるような励ましのコメントを含めてください。
"""