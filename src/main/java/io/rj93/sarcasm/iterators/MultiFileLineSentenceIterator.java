package io.rj93.sarcasm.iterators;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;

public class MultiFileLineSentenceIterator extends BaseSentenceIterator {
	
	private List<File> files;
	private int index = 0;
	private InputStream iStream;
	private LineIterator iter;

	public MultiFileLineSentenceIterator(List<File> files) throws IOException {
		for (File f : files){
			if (!f.exists() || !f.isFile())
				throw new FileNotFoundException("Unable to find file: " + f.getAbsolutePath());
		}
		this.files = files;
		iStream = new BufferedInputStream(new FileInputStream(files.get(index)));
		iter = IOUtils.lineIterator(iStream, "UTF-8");
	}

	public String nextSentence() {
		if (!iter.hasNext())
			getNewFile();
		
		String line = iter.nextLine();
        if (preProcessor != null)
            line = preProcessor.preProcess(line);
        
        return line;
	}

	public boolean hasNext() {
		if (!iter.hasNext())
			getNewFile();
		
		return iter.hasNext();
	}

	public void reset() {
		// TODO	
	}
	
	private void getNewFile() {
		if (index < files.size() - 1){
			try {
				index++;
				iStream = new BufferedInputStream(new FileInputStream(files.get(index)));
				iter = IOUtils.lineIterator(iStream, "UTF-8");
			} catch (IOException e) {
				System.err.println("Unable to open file: \'" + files.get(index) + "\'. Skipping file");
				getNewFile();
			}
		}
	}
	
	public boolean isPositive(){
		return !files.get(index).getName().contains("non-sarcy");
	}

}
