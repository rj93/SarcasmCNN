package io.rj93.sarcasm.cnn;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.SentimentChannel;

public class SentimentChannelTest {
	
	@Test
	public void test(){
		Channel channel = new SentimentChannel(3);
		
//		System.out.println(channel.getFeatureVector("bad test good"));
		
		List<String> strings = new ArrayList<String>();
		strings.add("bad test good");
		strings.add("bad bad bad");
		strings.add("good good good test good");
		System.out.println(channel.getFeatureVectors(strings).getFeatures());
	}
}
