from flask import Blueprint, request, jsonify
from app.models import ToxicityModel

bp = Blueprint('routes', __name__)

model = ToxicityModel('model/best_model_cnn.keras', 'model/tokenizer.pkl')

@bp.route('/api/check_toxicity', methods=['POST'])
def check_toxicity():
    try:
        data = request.get_json()
        comment = data.get('comment')
        if not comment:
            return jsonify({'error': 'Comment is required'}), 400

        # Analiza toksyczności
        toxicity_score = model.predict_toxicity(comment)
        return jsonify({'comment': comment, 'toxicity_score': toxicity_score}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
