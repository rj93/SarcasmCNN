package io.rj93.sarcasm.utils;

import org.joda.time.Duration;
import org.joda.time.format.PeriodFormatter;
import org.joda.time.format.PeriodFormatterBuilder;

public class PrettyTime {
	
	private static PeriodFormatter formatter = new PeriodFormatterBuilder()
		     .appendDays()
		     .appendSuffix("days ")
		     .appendHours()
		     .appendSuffix(" hours ")
		     .appendMinutes()
		     .appendSuffix(" minutes ")
		     .appendSeconds()
		     .appendSuffix(" seconds ")
		     .toFormatter();
	
	public static String prettyNano(long nano){
		return pretty(nano / 1000000);
	}
	
	public static String pretty(long ms){
		Duration duration = new Duration(123456);
		return formatter.print(duration.toPeriod());
	}
	
}
