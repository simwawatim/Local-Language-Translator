# translate_api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from transformers import MarianMTModel, MarianTokenizer

class TranslateView(APIView):
    def post(self, request):
        source_text = request.data.get("text")
        src_lang = request.data.get("src_lang", "en")
        tgt_lang = request.data.get("tgt_lang", "bem") 

        if not source_text:
            return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)

        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            translated = model.generate(**tokenizer(source_text, return_tensors="pt", padding=True))
            tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

            return Response({"translated_text": tgt_text})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
