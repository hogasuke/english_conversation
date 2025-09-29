import streamlit as st
import os
import time
from pathlib import Path
import wave
import pyaudio
from pydub import AudioSegment
from audiorecorder import audiorecorder
import numpy as np
from scipy.io.wavfile import write
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct

def record_audio(audio_input_file_path):
    """
    音声入力を受け取って音声ファイルを作成
    """

    audio = audiorecorder(
        start_prompt="発話開始",
        pause_prompt="やり直す",
        stop_prompt="発話終了",
        start_style={"color":"white", "background-color":"black"},
        pause_style={"color":"gray", "background-color":"white"},
        stop_style={"color":"white", "background-color":"black"}
    )

    if len(audio) > 0:
        audio.export(audio_input_file_path, format="wav")
    else:
        st.stop()

def transcribe_audio(audio_input_file_path):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
    """

    with open(audio_input_file_path, 'rb') as audio_input_file:
        transcript = st.session_state.openai_obj.audio.transcriptions.create(
            model="whisper-1",
            file=audio_input_file,
            language="en"
        )
    
    # 音声入力ファイルを削除
    os.remove(audio_input_file_path)

    return transcript

def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """

    temp_audio_output_filename = f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
        temp_audio_output_file.write(llm_response_audio)
    
    audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
    audio_mp3.export(audio_output_file_path, format="wav")

    # 音声出力用に一時的に作ったmp3ファイルを削除
    os.remove(temp_audio_output_filename)

def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """

    # 音声ファイルの読み込み
    audio = AudioSegment.from_wav(audio_output_file_path)
    
    # 速度を変更
    if speed != 1.0:
        # frame_rateを変更することで速度を調整
        modified_audio = audio._spawn(
            audio.raw_data, 
            overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
        modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

        modified_audio.export(audio_output_file_path, format="wav")

    # PyAudioで再生
    with wave.open(audio_output_file_path, 'rb') as play_target_file:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(play_target_file.getsampwidth()),
            channels=play_target_file.getnchannels(),
            rate=play_target_file.getframerate(),
            output=True
        )

        data = play_target_file.readframes(1024)
        while data:
            stream.write(data)
            data = play_target_file.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # LLMからの回答の音声ファイルを削除
    os.remove(audio_output_file_path)

def create_chain(system_template):
    """
    LLMによる回答生成用のChain作成
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt
    )

    return chain

def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    過去の会話履歴を考慮して、2回目以降は文脈に沿った問題を生成する
    """
    
    # 適応的難易度を取得
    difficulty = get_adaptive_difficulty()
    
    # 2回目以降は過去の会話履歴を考慮した問題文を生成
    if len(st.session_state.messages) > 0:
        # 文脈を考慮したChainを作成または使用（英語レベルと適応的難易度を考慮）
        base_template = ct.SYSTEM_TEMPLATE_CREATE_CONTEXTUAL_PROBLEM.format(
            english_level=st.session_state.englv
        )
        contextual_template = create_adaptive_problem_prompt(base_template, difficulty)
        
        if not hasattr(st.session_state, 'chain_create_contextual_problem') or \
           st.session_state.get('prev_english_level') != st.session_state.englv or \
           st.session_state.get('prev_difficulty') != difficulty:
            st.session_state.chain_create_contextual_problem = create_chain(contextual_template)
            st.session_state.prev_english_level = st.session_state.englv
            st.session_state.prev_difficulty = difficulty
        
        # 過去の会話履歴を要約して入力として使用
        recent_conversation = ""
        for message in st.session_state.messages[-6:]:  # 直近3回のやり取り（6メッセージ）を参照
            if message["role"] in ["user", "assistant"]:
                role_name = "User" if message["role"] == "user" else "AI"
                recent_conversation += f"{role_name}: {message['content']}\n"
        
        problem_input = f"Previous conversation context:\n{recent_conversation}\nUser's English level: {st.session_state.englv}\nGenerate a natural follow-up sentence:"
        problem = st.session_state.chain_create_contextual_problem.predict(input=problem_input)
    else:
        # 初回は通常の問題文生成（適応的難易度適用）
        adaptive_template = create_adaptive_problem_prompt(ct.SYSTEM_TEMPLATE_CREATE_PROBLEM, difficulty)
        initial_chain = create_chain(adaptive_template)
        problem = initial_chain.predict(input="")

    # LLMからの回答を音声データに変換
    llm_response_audio = st.session_state.openai_obj.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=problem
    )

    # 音声ファイルの作成
    audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
    save_to_wav(llm_response_audio.content, audio_output_file_path)

    # 音声ファイルの読み上げ
    play_wav(audio_output_file_path, st.session_state.speed)

    return problem, llm_response_audio

def create_evaluation():
    """
    ユーザー入力値の評価生成
    """

    llm_response_evaluation = st.session_state.chain_evaluation.predict(input="")

    return llm_response_evaluation

def initialize_learning_progress():
    """
    学習進捗データの初期化
    """
    if "learning_progress" not in st.session_state:
        st.session_state.learning_progress = ct.LEARNING_PROGRESS_KEYS.copy()

def extract_scores_from_evaluation(evaluation_text):
    """
    評価テキストから数値スコアを抽出
    Args:
        evaluation_text: LLMからの評価テキスト
    Returns:
        dict: 各種スコア
    """
    scores = {
        "accuracy": 0,
        "grammar": 0,
        "completeness": 0,
        "overall": 0
    }
    
    lines = evaluation_text.split('\n')
    for line in lines:
        if 'SCORE_ACCURACY:' in line:
            try:
                scores["accuracy"] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'SCORE_GRAMMAR:' in line:
            try:
                scores["grammar"] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'SCORE_COMPLETENESS:' in line:
            try:
                scores["completeness"] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'SCORE_OVERALL:' in line:
            try:
                scores["overall"] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
    
    return scores

def update_learning_progress(scores):
    """
    学習進捗を更新し、必要に応じてレベルアップ
    Args:
        scores: 評価スコア辞書
    """
    progress = st.session_state.learning_progress
    
    # 総回答数を増加
    progress["total_attempts"] += 1
    
    # 正答率の計算（overall scoreが70以上を成功とする）
    is_success = scores["overall"] >= 70
    if is_success:
        progress["correct_responses"] += 1
        progress["consecutive_success"] += 1
    else:
        progress["consecutive_success"] = 0
    
    # 正答率履歴の更新（直近10回）
    progress["accuracy_history"].append(is_success)
    if len(progress["accuracy_history"]) > 10:
        progress["accuracy_history"].pop(0)
    
    # レベルアップの判定
    check_level_up(scores)

def check_level_up(scores):
    """
    レベルアップ条件をチェックし、必要に応じてレベルを上げる
    Args:
        scores: 評価スコア辞書
    """
    progress = st.session_state.learning_progress
    
    # 連続成功でのレベルアップ
    if progress["consecutive_success"] >= progress["level_up_threshold"]:
        # 語彙レベルの調整
        if scores["accuracy"] >= 85 and progress["vocabulary_level"] < 10:
            progress["vocabulary_level"] += 1
            progress["consecutive_success"] = 0  # リセット
            
        # 文法レベルの調整
        if scores["grammar"] >= 80 and progress["grammar_level"] < 10:
            progress["grammar_level"] += 1
            progress["consecutive_success"] = 0  # リセット
            
        # 流暢さレベルの調整
        if scores["completeness"] >= 80 and progress["fluency_level"] < 10:
            progress["fluency_level"] += 1
            progress["consecutive_success"] = 0  # リセット
    
    # 長期的な成績が悪い場合のレベルダウン
    if len(progress["accuracy_history"]) >= 5:
        recent_success_rate = sum(progress["accuracy_history"][-5:]) / 5
        if recent_success_rate < 0.3:  # 30%未満の成功率
            if progress["vocabulary_level"] > 1:
                progress["vocabulary_level"] = max(1, progress["vocabulary_level"] - 1)
            if progress["grammar_level"] > 1:
                progress["grammar_level"] = max(1, progress["grammar_level"] - 1)
            if progress["fluency_level"] > 1:
                progress["fluency_level"] = max(1, progress["fluency_level"] - 1)

def get_adaptive_difficulty():
    """
    現在の学習レベルに基づいて適応的な難易度を取得
    Returns:
        dict: 難易度設定
    """
    progress = st.session_state.learning_progress
    
    # 各レベルの平均を計算
    avg_level = (progress["vocabulary_level"] + progress["grammar_level"] + progress["fluency_level"]) // 3
    avg_level = max(1, min(10, avg_level))  # 1-10の範囲に制限
    
    return ct.DIFFICULTY_LEVELS.get(avg_level, ct.DIFFICULTY_LEVELS[1])

def create_adaptive_problem_prompt(base_template, difficulty):
    """
    適応的難易度を考慮した問題生成プロンプトを作成
    Args:
        base_template: ベースとなるプロンプトテンプレート
        difficulty: 難易度設定
    Returns:
        str: 調整されたプロンプトテンプレート
    """
    adaptive_prompt = base_template + f"""

    【適応的難易度調整】
    - 語彙レベル: {difficulty['vocabulary']}
    - 文法構造: {difficulty['grammar']}
    - 文長: {difficulty['sentence_length']} words
    
    上記の難易度に従って、ユーザーの現在のレベルに適した問題を生成してください。
    """
    
    return adaptive_prompt