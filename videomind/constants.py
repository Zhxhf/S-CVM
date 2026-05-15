# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

IGNORE_INDEX = -100

REG_TOKEN = '<|reg|>'

SEG_S_TOKEN = '<|seg_start|>'
SEG_E_TOKEN = '<|seg_end|>'

PLANNER_PROMPT = (
    'You are acting as the planner now. '
    'Given a question about the video, your task is to analyze the question and identify the best way to answer this question. '
    'You have access to the following tools:\n\n'
    'Grounder: Accepts a text query and localize the relevant video segment according to the query.\n'
    'Verifier: A tool supporting grounder by verifying the reliability of its outputs.\n'
    'Answerer: Answer a given question directly based on the whole video or a cropped video segment.\n\n'
    'Your response must be a list in JSON format. '
    'A valid plan for reasoning could be "grounder, verifier, answer", "grounder, verifier", or "answerer", depending on the given question. '
    'Please see an example for the format below.\n\n'
    '[{{"type": "grounder", "value": "<text query>"}}, {{"type": "verifier"}}, {{"type": "answerer"}}]\n\n'
    'Note that only the grounder can accept an argument called "value", which is the text query used for grounding. '
    "Now I give you the question: '{}'. "
    'Please think carefully and respond with your plan in JSON directly.')

GROUNDER_PROMPT = (
    'You are acting as the grounder now. '
    'Given a video and a text query, your goal is to temporally localize the video moment described by the query. '
    'If the query is directly describing a moment, simply localize it according to its content. '
    "Otherwise, if the moment is described as 'before/after a pivotal event', you need to determine the actual event it refers to. "
    'The localized moment should only cover the target event. '
    "Now I give you the query: '{}'. "
    'Please think carefully and provide your response.')

VERIFIER_PROMPT = (
    'You are acting as the verifier now. '
    'You will be presented a text query describing a moment that potentialy happens in the given video. '
    f'Your task is to identify whether the video segment between {SEG_S_TOKEN} and {SEG_E_TOKEN} perfectly covers the moment. '
    f'If the described moment can be seen in the video, please focus on verifying whether the moment starts at {SEG_S_TOKEN} and ends at {SEG_E_TOKEN}. '
    "Respond with 'Yes' if you think the moment boundaries are correct, otherwise 'No'. "
    "If the described moment cannot be seen in the video, respond with 'No' directly. "
    "Now I give you the query: '{}'. "
    "Please think carefully and respond with 'Yes' or 'No' directly.")


REVERSE_QUESTION_PROMPT = (
    'You are acting as an evidence-faithfulness probe now. '
    'You are given a video segment together with an answer produced from that segment. '
    'Your task is to reconstruct the most likely original question that this answer is addressing under the current evidence. '
    'Please respond with a single concise question only. '
    "Now I give you the answer: '{}'. "
    'Please reconstruct the implied question directly.' )


MULTI_SEGMENT_ANSWER_PROMPT = (
    'You are acting as the answerer now. '
    'You are given two complementary video clips, Clip A and Clip B, sampled from the same original video. '
    'Use both clips jointly to answer the question. '
    'If one clip provides context and the other provides the key event, combine them carefully instead of relying on only one clip. '
    'Now I give you the question: "{}". '
    'Please answer using the combined evidence from Clip A and Clip B.' )

FEEDBACK_PROMPT = (
    'You are acting as a feedback policy module for long-video question answering. '
    'Given the original question, the current answer, and the reconstructed question implied by the answer, '
    'identify the main missing information that prevents a faithful answer. '
    'Choose exactly one label from the following list and output the label only: '
    '[subject, action, local_context, reason, result, temporal_order, dialogue, uncertain]. '
    'Question: {}\n'
    'Current answer: {}\n'
    'Reconstructed question: {}\n'
    'Output one label only.' )
