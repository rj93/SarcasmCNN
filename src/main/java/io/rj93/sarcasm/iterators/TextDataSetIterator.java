package io.rj93.sarcasm.iterators;

import java.util.List;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class TextDataSetIterator implements DataSetIterator {

	public boolean hasNext() {
		// TODO Auto-generated method stub
		return false;
	}

	public DataSet next() {
		// TODO Auto-generated method stub
		return null;
	}

	public DataSet next(int num) {
		// TODO Auto-generated method stub
		return null;
	}

	public int totalExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int inputColumns() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return 0;
	}

	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public void reset() {
		// TODO Auto-generated method stub

	}

	public int batch() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int cursor() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int numExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub

	}

	public DataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return null;
	}

	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return null;
	}

}
