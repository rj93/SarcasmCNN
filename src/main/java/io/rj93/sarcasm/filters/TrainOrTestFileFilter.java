package io.rj93.sarcasm.filters;

import java.io.File;
import java.io.FileFilter;

public class TrainOrTestFileFilter implements FileFilter {
	
	private boolean isTraining;
	private String name;
	
	public TrainOrTestFileFilter(boolean isTraining){
		this.isTraining = isTraining;
		if (isTraining)
			name = "train";
		else
			name = "test";
	}
	
	@Override
	public boolean accept(File file) {
		return file.getParentFile().getParentFile().getName().equals(name);
	}

}
