from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env laden
load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini API konfigurieren
api_key = os.getenv('GEMINI_KEY', '')
genai.configure(api_key=api_key)

# Gemini Model initialisieren
model = genai.GenerativeModel('gemini-2.0-flash')

# Sprach-Mapping für bessere Prompts
LANGUAGE_NAMES = {
    'de-DE': 'Deutsch',
    'en-US': 'Amerikanisches Englisch',
    'en-GB': 'Britisches Englisch',
    'fr-FR': 'Französisch',
    'es-ES': 'Spanisch',
    'it-IT': 'Italienisch',
    'pt-PT': 'Portugiesisch',
    'nl-NL': 'Niederländisch',
    'pl-PL': 'Polnisch',
    'sv-SE': 'Schwedisch'
}

@app.route('/api/transcribe-ipa', methods=['POST'])
def transcribe_to_ipa():
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'de-DE')
        
        if not text:
            return jsonify({'error': 'Kein Text angegeben'}), 400
        
        language_name = LANGUAGE_NAMES.get(language, 'Deutsch')
        
        prompt = f"""Wandle den folgenden {language_name}en Satz in die Internationale Phonetische Alphabet (IPA) Notation um.
        
Satz: "{text}"

Antworte NUR mit der IPA-Transkription, ohne zusätzliche Erklärungen oder Text. 
Verwende eckige Klammern [ ] für die phonetische Transkription.
Achte auf die korrekte Aussprache für {language_name}."""

        response = model.generate_content(prompt)
        ipa_text = response.text.strip()
        
        return jsonify({
            'success': True,
            'ipa': ipa_text,
            'original': text,
            'language': language
        })
        
    except Exception as e:
        print(f"Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/find-similar-sounding', methods=['POST'])
def find_similar_sounding():
    try:
        data = request.json
        ipa = data.get('ipa', '')
        original_text = data.get('originalText', '')
        source_language = data.get('sourceLanguage', 'de-DE')
        target_language = data.get('targetLanguage', 'en-US')
        
        if not ipa and not original_text:
            return jsonify({'error': 'Kein IPA oder Text angegeben'}), 400
        
        source_lang_name = LANGUAGE_NAMES.get(source_language, 'Deutsch')
        target_lang_name = LANGUAGE_NAMES.get(target_language, 'Englisch')
        
        prompt = f"""Du bist ein Experte für Phonetik und Sprachen.

Originaltext ({source_lang_name}): "{original_text}"
IPA-Transkription: {ipa}

Finde einen Satz oder eine Phrase auf {target_lang_name}, die möglichst ÄHNLICH KLINGT wie der Originaltext.
Es geht NICHT um eine Übersetzung des Inhalts, sondern um einen Satz der phonetisch ähnlich klingt - also ähnliche Laute und Rhythmus hat.

Beispiel: "Ich liebe dich" auf Deutsch könnte auf Englisch ähnlich klingen wie "Ish leave a dish" (nicht die Übersetzung "I love you").

Antworte NUR mit dem ähnlich klingenden Satz auf {target_lang_name}, ohne Erklärungen."""

        response = model.generate_content(prompt)
        similar_text = response.text.strip()
        
        return jsonify({
            'success': True,
            'similarText': similar_text,
            'originalText': original_text,
            'ipa': ipa,
            'targetLanguage': target_language
        })
        
    except Exception as e:
        print(f"Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate-back', methods=['POST'])
def translate_back():
    try:
        data = request.json
        text = data.get('text', '')
        source_language = data.get('sourceLanguage', 'en-US')
        target_language = data.get('targetLanguage', 'de-DE')
        
        if not text:
            return jsonify({'error': 'Kein Text angegeben'}), 400
        
        source_lang_name = LANGUAGE_NAMES.get(source_language, 'Englisch')
        target_lang_name = LANGUAGE_NAMES.get(target_language, 'Deutsch')
        
        prompt = f"""Übersetze den folgenden Satz von {source_lang_name} nach {target_lang_name}.

Satz: "{text}"

Antworte NUR mit der Übersetzung, ohne zusätzliche Erklärungen."""

        response = model.generate_content(prompt)
        translation = response.text.strip()
        
        return jsonify({
            'success': True,
            'translation': translation,
            'originalText': text,
            'sourceLanguage': source_language,
            'targetLanguage': target_language
        })
        
    except Exception as e:
        print(f"Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Server startet auf http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
