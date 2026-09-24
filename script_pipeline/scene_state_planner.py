"""Qwen proposes operations; the deterministic store validates them before commit."""
import json
import urllib.request

from script_pipeline.world_store import apply_operations


def propose_event(state, text, event_id, source_unit_id, model='qwen2.5:32b-instruct-q4_K_M'):
    prompt = '''Return only JSON {"operations": [...]}. You annotate a persistent film scene.
Do not invent entities or change dialogue. Available operations:
{"op":"transfer","entity_id":"PROP_ID","from":"CURRENT_OWNER","to":"NEW_OWNER","socket":"left_hand"}
or {"op":"set","entity_id":"ID","field":"position|yaw|pose|wardrobe_id|present","expected":CURRENT_VALUE,"value":NEW_VALUE}.
Only express the explicitly requested action. Use exact existing entity IDs. Coordinates are Z-up metres.
STATE: ''' + json.dumps(state) + '\nACTION: ' + text
    payload = {'model': model, 'prompt': prompt, 'stream': False, 'format': 'json',
               'options': {'temperature': 0, 'num_predict': 600, 'num_ctx': 4096}, 'keep_alive': 0}
    request = urllib.request.Request('http://127.0.0.1:11434/api/generate',
                                     data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.load(response)
    proposed = json.loads(raw['response'])
    event = {'id': event_id, 'source_unit_id': source_unit_id, 'text': text,
             'model': model, 'operations': proposed['operations']}
    apply_operations(state, event['operations'])
    return event
