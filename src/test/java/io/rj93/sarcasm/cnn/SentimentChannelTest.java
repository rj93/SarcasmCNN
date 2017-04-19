package io.rj93.sarcasm.cnn;

import org.junit.Test;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.SentimentChannel;

public class SentimentChannelTest {
	
	@Test
	public void test(){
		Channel channel = new SentimentChannel(3);
		
		System.out.println(channel.getFeatureVector("bad test good"));
	}
}
