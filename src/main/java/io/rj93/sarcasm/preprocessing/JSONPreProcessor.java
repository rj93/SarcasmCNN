package io.rj93.sarcasm.preprocessing;

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.json.JSONObject;

public class JSONPreProcessor extends TextPreProcessor implements SentencePreProcessor {

	public String preProcess(String json) {
		JSONObject comment = new JSONObject(json);
		return super.preProcess(comment.getString("body"));
	}

}
