package io.rj93.sarcasm.utils.filters;

import java.io.File;
import java.io.FileFilter;

public class TrainOrTestFileFilter implements FileFilter {
	
	private boolean isTraining;
	private String name;
	private int parentIndex;
	
	public TrainOrTestFileFilter(boolean isTraining, int parentIndex){
		this.isTraining = isTraining;
		if (isTraining)
			name = "train";
		else
			name = "test";
		
		this.parentIndex = parentIndex;
	}
	
	@Override
	public boolean accept(File file) {
		// get correct parent directory
		File parent = file.getParentFile();
		for (int i = 1; i < parentIndex; i++){
			parent = parent.getParentFile();
		}
		return parent.getName().equals(name);
	}

}
