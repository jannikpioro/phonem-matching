#!/usr/bin/env python3
"""
Filtert Wortpaare, um nur ECHTE englische Wörter zu behalten.
Nutzt eine Liste der häufigsten englischen Wörter.
"""

import json
import argparse

# Die ~3000 häufigsten englischen Wörter (Auszug - die wichtigsten)
# Quelle: Kombiniert aus verschiedenen Frequenzlisten
COMMON_ENGLISH_WORDS = {
    # Artikel, Pronomen, Präpositionen
    'a', 'an', 'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'who', 'what', 'which', 'where', 'when', 'why', 'how',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'out', 'up', 'down',
    
    # Verben (häufig)
    'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'done', 'doing',
    'say', 'says', 'said', 'saying',
    'go', 'goes', 'went', 'gone', 'going',
    'get', 'gets', 'got', 'getting',
    'make', 'makes', 'made', 'making',
    'know', 'knows', 'knew', 'known', 'knowing',
    'think', 'thinks', 'thought', 'thinking',
    'take', 'takes', 'took', 'taken', 'taking',
    'see', 'sees', 'saw', 'seen', 'seeing',
    'come', 'comes', 'came', 'coming',
    'want', 'wants', 'wanted', 'wanting',
    'use', 'uses', 'used', 'using',
    'find', 'finds', 'found', 'finding',
    'give', 'gives', 'gave', 'given', 'giving',
    'tell', 'tells', 'told', 'telling',
    'work', 'works', 'worked', 'working',
    'call', 'calls', 'called', 'calling',
    'try', 'tries', 'tried', 'trying',
    'ask', 'asks', 'asked', 'asking',
    'need', 'needs', 'needed', 'needing',
    'feel', 'feels', 'felt', 'feeling',
    'become', 'becomes', 'became', 'becoming',
    'leave', 'leaves', 'left', 'leaving',
    'put', 'puts', 'putting',
    'mean', 'means', 'meant', 'meaning',
    'keep', 'keeps', 'kept', 'keeping',
    'let', 'lets', 'letting',
    'begin', 'begins', 'began', 'begun', 'beginning',
    'seem', 'seems', 'seemed', 'seeming',
    'help', 'helps', 'helped', 'helping',
    'show', 'shows', 'showed', 'shown', 'showing',
    'hear', 'hears', 'heard', 'hearing',
    'play', 'plays', 'played', 'playing',
    'run', 'runs', 'ran', 'running',
    'move', 'moves', 'moved', 'moving',
    'live', 'lives', 'lived', 'living',
    'believe', 'believes', 'believed', 'believing',
    'hold', 'holds', 'held', 'holding',
    'bring', 'brings', 'brought', 'bringing',
    'write', 'writes', 'wrote', 'written', 'writing',
    'stand', 'stands', 'stood', 'standing',
    'sit', 'sits', 'sat', 'sitting',
    'lose', 'loses', 'lost', 'losing',
    'pay', 'pays', 'paid', 'paying',
    'meet', 'meets', 'met', 'meeting',
    'include', 'includes', 'included', 'including',
    'continue', 'continues', 'continued', 'continuing',
    'set', 'sets', 'setting',
    'learn', 'learns', 'learned', 'learning',
    'change', 'changes', 'changed', 'changing',
    'lead', 'leads', 'led', 'leading',
    'understand', 'understands', 'understood', 'understanding',
    'watch', 'watches', 'watched', 'watching',
    'follow', 'follows', 'followed', 'following',
    'stop', 'stops', 'stopped', 'stopping',
    'create', 'creates', 'created', 'creating',
    'speak', 'speaks', 'spoke', 'spoken', 'speaking',
    'read', 'reads', 'reading',
    'spend', 'spends', 'spent', 'spending',
    'grow', 'grows', 'grew', 'grown', 'growing',
    'open', 'opens', 'opened', 'opening',
    'walk', 'walks', 'walked', 'walking',
    'win', 'wins', 'won', 'winning',
    'teach', 'teaches', 'taught', 'teaching',
    'offer', 'offers', 'offered', 'offering',
    'remember', 'remembers', 'remembered', 'remembering',
    'consider', 'considers', 'considered', 'considering',
    'appear', 'appears', 'appeared', 'appearing',
    'buy', 'buys', 'bought', 'buying',
    'wait', 'waits', 'waited', 'waiting',
    'serve', 'serves', 'served', 'serving',
    'die', 'dies', 'died', 'dying',
    'send', 'sends', 'sent', 'sending',
    'build', 'builds', 'built', 'building',
    'stay', 'stays', 'stayed', 'staying',
    'fall', 'falls', 'fell', 'fallen', 'falling',
    'cut', 'cuts', 'cutting',
    'reach', 'reaches', 'reached', 'reaching',
    'kill', 'kills', 'killed', 'killing',
    'raise', 'raises', 'raised', 'raising',
    'pass', 'passes', 'passed', 'passing',
    'sell', 'sells', 'sold', 'selling',
    'decide', 'decides', 'decided', 'deciding',
    'return', 'returns', 'returned', 'returning',
    'explain', 'explains', 'explained', 'explaining',
    'hope', 'hopes', 'hoped', 'hoping',
    'develop', 'develops', 'developed', 'developing',
    'carry', 'carries', 'carried', 'carrying',
    'break', 'breaks', 'broke', 'broken', 'breaking',
    'receive', 'receives', 'received', 'receiving',
    'agree', 'agrees', 'agreed', 'agreeing',
    'support', 'supports', 'supported', 'supporting',
    'hit', 'hits', 'hitting',
    'produce', 'produces', 'produced', 'producing',
    'eat', 'eats', 'ate', 'eaten', 'eating',
    'cover', 'covers', 'covered', 'covering',
    'catch', 'catches', 'caught', 'catching',
    'draw', 'draws', 'drew', 'drawn', 'drawing',
    'choose', 'chooses', 'chose', 'chosen', 'choosing',
    'light', 'lights', 'lit', 'lighting',
    'shout', 'shouts', 'shouted', 'shouting',
    'finish', 'finishes', 'finished', 'finishing',
    'pick', 'picks', 'picked', 'picking',
    'house', 'houses', 'housed', 'housing',
    'guest', 'guests',
    'march', 'marches', 'marched', 'marching',
    'order', 'orders', 'ordered', 'ordering',
    'sprint', 'sprints', 'sprinted', 'sprinting',
    'track', 'tracks', 'tracked', 'tracking',
    'rank', 'ranks', 'ranked', 'ranking',
    'dine', 'dines', 'dined', 'dining', 'diner',
    'shrank', 'shrink', 'shrinks', 'shrunk',
    'balk', 'balks', 'balked', 'balking',
    'lock', 'locks', 'locked', 'locking',
    'peak', 'peaks', 'peaked', 'peaking',
    'flash', 'flashes', 'flashed', 'flashing',
    'strict', 'stricter', 'strictest',
    
    # Substantive (häufig)
    'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'children',
    'world', 'life', 'hand', 'part', 'place', 'case', 'week', 'company', 'system',
    'program', 'question', 'work', 'government', 'number', 'night', 'point', 'home',
    'water', 'room', 'mother', 'area', 'money', 'story', 'fact', 'month', 'lot',
    'right', 'study', 'book', 'eye', 'job', 'word', 'business', 'issue', 'side',
    'kind', 'head', 'house', 'service', 'friend', 'father', 'power', 'hour', 'game',
    'line', 'end', 'member', 'law', 'car', 'city', 'community', 'name', 'president',
    'team', 'minute', 'idea', 'kid', 'body', 'information', 'back', 'parent', 'face',
    'others', 'level', 'office', 'door', 'health', 'person', 'art', 'war', 'history',
    'party', 'result', 'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment',
    'air', 'teacher', 'force', 'education', 'foot', 'boy', 'age', 'policy', 'process',
    'music', 'market', 'sense', 'nation', 'plan', 'college', 'interest', 'death', 'experience',
    'effect', 'use', 'class', 'control', 'care', 'field', 'development', 'role', 'effort',
    'rate', 'heart', 'drug', 'show', 'leader', 'light', 'voice', 'wife', 'police',
    'mind', 'difference', 'period', 'value', 'building', 'action', 'authority', 'model',
    'fish', 'future', 'wrong', 'size', 'form', 'food', 'practice', 'cost', 'bank',
    'view', 'fire', 'society', 'tax', 'player', 'agreement', 'support', 'event', 'official',
    'center', 'news', 'staff', 'range', 'focus', 'trade', 'source', 'loss', 'rock',
    'fear', 'oil', 'blood', 'road', 'truth', 'sun', 'paper', 'analysis', 'growth',
    'love', 'pain', 'hope', 'song', 'star', 'tree', 'dream', 'peace', 'rain',
    'gift', 'beast', 'guest', 'marsh', 'diner', 'villa', 'cannon', 'villain',
    'builder', 'loiter', 'armor', 'felon', 'demon', 'climate', 'balance',
    'strict', 'distinct', 'instinct', 'frost', 'craft', 'shaft', 'draft',
    
    # Adjektive (häufig)
    'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
    'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
    'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'hot', 'cold',
    'free', 'full', 'sure', 'clear', 'true', 'whole', 'real', 'hard', 'best',
    'better', 'short', 'low', 'late', 'general', 'human', 'local', 'black', 'white',
    'red', 'blue', 'green', 'dark', 'bright', 'light', 'strong', 'weak', 'fast',
    'slow', 'happy', 'sad', 'angry', 'kind', 'nice', 'fine', 'open', 'close',
    'dead', 'alive', 'wild', 'quiet', 'loud', 'soft', 'rough', 'smooth', 'wet', 'dry',
    'finnish', 'strict', 'distinct',
    
    # Adverbien
    'up', 'so', 'out', 'just', 'now', 'how', 'then', 'more', 'also', 'here',
    'well', 'only', 'very', 'even', 'back', 'there', 'down', 'still', 'too',
    'much', 'again', 'never', 'always', 'often', 'once', 'soon', 'ever', 'yet',
    
    # Zahlen
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'hundred', 'thousand', 'million',
    
    # Weitere interessante Wörter für unseren Zweck
    'light', 'night', 'fight', 'right', 'might', 'sight', 'tight', 'bright',
    'house', 'mouse', 'louse',
    'round', 'sound', 'found', 'ground', 'bound', 'hound', 'mound', 'pound', 'wound',
    'shout', 'about', 'scout', 'stout', 'trout', 'clout', 'grout', 'snout', 'spout',
    'beast', 'feast', 'least', 'yeast', 'east', 'west', 'best', 'rest', 'test', 'nest',
    'guest', 'quest', 'chest', 'crest', 'pressed', 'dressed', 'guessed', 'blessed',
    'marsh', 'harsh', 'cash', 'flash', 'crash', 'trash', 'splash', 'clash', 'dash',
    'strict', 'trick', 'stick', 'thick', 'quick', 'brick', 'click', 'flick', 'slick',
    'track', 'black', 'crack', 'stack', 'snack', 'attack', 'pack', 'lack', 'back',
    'rank', 'bank', 'tank', 'blank', 'thank', 'frank', 'plank', 'drank', 'shrank',
    'sprint', 'print', 'hint', 'mint', 'tint', 'flint', 'stint', 'glint', 'squint',
    'gift', 'lift', 'drift', 'shift', 'swift', 'thrift', 'rift',
    'order', 'border', 'murder', 'disorder', 'recorder',
    'villa', 'gorilla', 'vanilla', 'umbrella', 'cinderella',
    'cannon', 'canyon', 'abandon', 'companion', 'champion',
    'villain', 'mountain', 'fountain', 'captain', 'curtain', 'certain', 'contain',
    'builder', 'wilder', 'milder', 'holder', 'folder', 'shoulder', 'boulder',
    'armor', 'harbor', 'labor', 'neighbor', 'favor', 'flavor', 'manor', 'manor',
    'felon', 'melon', 'talon', 'gallon', 'salon', 'falcon', 'beacon',
    'demon', 'lemon', 'sermon', 'summon', 'common', 'salmon',
    'climate', 'primate', 'imate', 'animate', 'estimate',
    'balance', 'advance', 'romance', 'finance', 'distance', 'instance', 'substance',
    'loiter', 'pointer', 'printer', 'winter', 'splinter', 'center', 'enter',
    'craft', 'draft', 'shaft', 'raft', 'daft', 'after', 'laughter',
    'frost', 'cost', 'lost', 'toss', 'boss', 'cross', 'moss', 'gloss',
    'tuna', 'luna', 'sauna',
    'diner', 'miner', 'finer', 'liner', 'shiner', 'whiner',
    'picks', 'kicks', 'tricks', 'sticks', 'clicks', 'flicks', 'bricks',
    'locks', 'rocks', 'socks', 'blocks', 'clocks', 'stocks', 'knocks', 'shocks',
    'peaks', 'leaks', 'speaks', 'weeks', 'seeks', 'creeks', 'cheeks', 'freaks',
    'flash', 'slash', 'clash', 'crash', 'trash', 'smash', 'splash', 'stash', 'bash',
    'flesh', 'fresh', 'mesh', 'thresh',
    'shrift', 'drift', 'gift', 'lift', 'rift', 'shift', 'swift', 'thrift',
    'bliss', 'kiss', 'miss', 'hiss', 'this', 'abyss', 'dismiss',
    'rife', 'life', 'wife', 'knife', 'strife',
    'balked', 'walked', 'talked', 'stalked', 'chalked',
    'lit', 'hit', 'sit', 'bit', 'fit', 'kit', 'pit', 'wit', 'split', 'quit', 'spit',
    'halt', 'salt', 'malt', 'fault', 'vault', 'assault', 'default',
    'heart', 'start', 'part', 'art', 'cart', 'chart', 'dart', 'smart', 'tart',
    'lent', 'rent', 'sent', 'bent', 'dent', 'tent', 'went', 'spent', 'scent', 'vent',
    'brow', 'row', 'flow', 'glow', 'grow', 'know', 'show', 'slow', 'snow', 'throw',
    'troy', 'boy', 'joy', 'toy', 'annoy', 'deploy', 'destroy', 'employ', 'enjoy',
    'grout', 'scout', 'shout', 'snout', 'spout', 'stout', 'trout', 'clout', 'about',
}

def load_pairs(filepath: str) -> list:
    """Lädt Wortpaare aus JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('top_pairs', [])


def is_real_english_word(word: str) -> bool:
    """Prüft ob das Wort ein echtes englisches Wort ist."""
    word_lower = word.lower().strip("'")
    return word_lower in COMMON_ENGLISH_WORDS


def filter_pairs(pairs: list) -> list:
    """Filtert nur Paare mit echten englischen Wörtern."""
    filtered = []
    for p in pairs:
        english_word = p['target']['word']
        if is_real_english_word(english_word):
            filtered.append(p)
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Filtere nur echte englische Wörter')
    parser.add_argument('--input', '-i', default='all_good_pairs.json', help='Eingabedatei')
    parser.add_argument('--output', '-o', default='real_english_pairs.json', help='Ausgabedatei')
    
    args = parser.parse_args()
    
    pairs = load_pairs(args.input)
    print(f"Geladen: {len(pairs)} Paare")
    
    filtered = filter_pairs(pairs)
    print(f"Nach Filter (echte englische Wörter): {len(filtered)} Paare")
    
    # Speichern
    output_data = {
        'filtered_count': len(filtered),
        'top_pairs': filtered
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nGespeichert in: {args.output}")
    
    # Zeige Beispiele
    print(f"\nTop 30 Paare mit echten englischen Wörtern:")
    print("="*60)
    for i, p in enumerate(filtered[:30], 1):
        de = p['source']['word']
        en = p['target']['word']
        sim = p['similarity']
        print(f"{i:2}. {sim:.0%}  {de:<20} ↔ {en:<20}")


if __name__ == '__main__':
    main()
