using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics;

public class TrainingScript : MonoBehaviour {

	[SerializeField] Text debug; 
	string[,,] data;		
	double[,,] doubleData;
	double[,] density;
	int[,,] intData;
	int nClass;
	string file;
	
	/* parameters for training methods */
	int[,] N; //parameter for Binomial	
	double[,] std; // parameter for Gaussian
	double[,] mean; // parameter for Gaussian
	double[,] alpha; // parameter for Gamma
	double[,] beta; // parameter for Gamma
	double[,] probability; // parameter for Binomial
	double[,] lambda; //parameter for Poisson and Exponential
	double[,,,] pertinences; //parameter for Fuzzy (Zadeh) method
	double[] weights;

	// Use this for initialization
	void Start () {
		nClass = 3;
		file = "Assets/DB/bancoGam200flins.csv";
		readDB();		
		
		//convertToInt(); //fills intData
		//trainingFBinNB(); //fills parameters for Binomial
		//assessingFBinNB(); //fills density for Binomial
		
		//trainingFPoiNB(); //fills parameters for Poisson
		//assessingFPoiNB(); //fills density for Poisson
		
		//convertToDouble(); //fills doubleData
		//trainingFExpNB();
		//assessingFExpNB(); 
		
		//trainingFGauNB();
		//assessingFGauNB(); 
		
		//trainingFGamNB();
		//assessingFGamNB(); 
		
		//assessment(); //create the confusion matrix
		
		/** \/ Gamma Ponderada \/ **/
		convertToDouble();
		trainingFGamNB();
		weights = new double[3];
		
		double c1 = 0.8, c2 = 0.1, c3 = 0.1;
		weights[0] = c1;
		weights[1] = c2;
		weights[2] = c3;
		Debug.Log("Pesos: " + weights[0] + " " + weights[1] + " " + weights[2]);
		assessingWFGamNB();		
		assessment(); //create the confusion matrix
			
		for(double i = 0.7; i > 0; i-=0.01){
			double n = 0.8 - i + 0.1;
			c1 = i;
			c2 = n;
			c3 = 0.1;
			weights[0] = c1;
			weights[1] = c2;
			weights[2] = c3;
			Debug.Log("Pesos: " + weights[0] + " " + weights[1] + " " + weights[2]);
			assessingWFGamNB();		
			assessment(); //create the confusion matrix
			for(double j = n - 0.1; j > 0; j-=0.01){
				c2 = j; c3 = 1 - c2 - c1;
				weights[0] = c1;
				weights[1] = c2;
				weights[2] = c3;
				Debug.Log("Pesos: " + weights[0] + " " + weights[1] + " " + weights[2]);
				assessingWFGamNB();		
				assessment(); //create the confusion matrix
			}
		}
		
		
		/*for(int i = 0; i < 100; i++){
			weights = new double[3];
			weights[0] = c1;
			weights[1] = c2;
			weights[2] = c3;
		
			convertToDouble();
			trainingFGamNB();
			assessingWFGamNB();		
			assessment(); //create the confusion matrix
			
			if(i)
			c1 -= 0.1; c2 +=0.1; c3 = 1 - c1 - c2;
		}*/
				
	}
	
	// Update is called once per frame
	void Update () {
		
	}
		
	/* METHODS FOR DISCRETE DATA */
	/**
	*
	* Method that converts the data recovered from the text file for discrete 
	* classifiers.
	*
	**/
	void convertToInt(){
		intData = new int[data.GetLength(0), data.GetLength(1), data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int col = 0; col < data.GetLength(2); col++)
					intData[cl, line, col] = Convert.ToInt32(data[cl, line, col]);				
	}
			
	/**
	*
	* Method that calculates the pertinence function for the fuzzy probability.
	* Discrete data only. [Zadeh, 1968]
	*
	**/	
	void calcPertinencesInt(){
		//1st - calculate sturges
		int sturges = Convert.ToInt32(Math.Ceiling(1 + (3.22 * Math.Log10(intData.GetLength(1)))));
		pertinences = new double[intData.GetLength(0),sturges,intData.GetLength(2),3];

		for (int cl = 0; cl < intData.GetLength(0); ++cl) {
			for (int dim = 0; dim < intData.GetLength(2); ++dim) {
				//2nd - get max and min
				double max = intData[cl,0,dim];
				double min = intData[cl,0,dim];
				
				for (int j = 1; j < intData.GetLength(1); ++j) {
					if (max < intData[cl,j,dim])
						max = intData[cl,j,dim];
					if (min > intData[cl,j,dim])
						min = intData[cl,j,dim];
				}

				//3rd - calculate frequencies
				double[,] freq = new double[sturges,3];
				double step = (max - min) / sturges;

				for (int l = 0; l < sturges; ++l) {
					freq[l,0] = min + step * l;
					freq[l,1] = freq[l,0] + step;
				}

				for(int line = 0; line < intData.GetLength(1); ++line){
					for (int st = 0; st < sturges; ++st) {
						if (st == sturges - 1) {
							if (intData[cl,line,dim] >= freq[st,0] && intData[cl,line,dim] <= freq[st,1])
								freq[st,2] += 1;
						} else if (intData[cl,line,dim] >= freq[st,0] && intData[cl,line,dim] < freq[st,1]) {
							freq[st,2] += 1;
						}
					}
				}

				//4th - calculate relative frequencies
				//for (int st = 0; st < sturges; ++st)
					//freq[st,2] /= intData.GetLength(1);

				//5th - calculate pertinences
				double maxFreq = 0;
				for (int st = 1; st < sturges; ++st)
					if (maxFreq < freq[st,2])
						maxFreq = freq[st,2];
					
				for (int st = 0; st < sturges; ++st) {
					pertinences[cl,st,dim,0] = freq[st,0];
					pertinences[cl,st,dim,1] = freq[st,1];
					if (freq[st,2] == 0.0)
						pertinences[cl,st,dim,2] = 0.001;
					else
						pertinences[cl,st,dim,2] = freq[st,2]/maxFreq;
				}
			}
		}
		
		/*String matrix = "Pertinences:\n";				
		for(int i = 0; i < 3; i++){
			matrix += "C" + (i + 1) + "\n";
			for(int j = 0; j < 3; j++){
				matrix += "D" + (j + 1) + ": \n";
				for(int k = 0; k < pertinences.GetLength(1); k++){
				matrix += "[" + pertinences[i,k,j,0] + " - " + pertinences[i,k,j,1] + 
							"]: " + pertinences[i,k,j,2] + "\n";
			}
		}
		}
		
		Debug.Log(matrix);*/
	}
		
	/**
	*
	* Method that returns the pertinence for the specific value from the data.
	* Discrete data only.
	*
	**/	
	double getLogPertinenceInt(int cl, int auxCl, int line, int dim){
		for (int st = 0; st < pertinences.GetLength(1); ++st) {
			if (st == pertinences.GetLength(1) - 1) {
				if (intData[auxCl,line,dim] >= pertinences[cl,st,dim,0] &&
					intData[auxCl,line,dim] <= pertinences[cl,st,dim,1])
					return Math.Log(pertinences[cl,st,dim,2]);
			} else if (intData[auxCl,line,dim] >= pertinences[cl,st,dim,0] &&
				intData[auxCl,line,dim] < pertinences[cl,st,dim,1]){
				return Math.Log(pertinences[cl,st,dim,2]);
			}
		}
		return 0.0;
	}
	
	/**
	*
	* Method that estimates parameters for the Fuzzy Binomial Naive Bayes 
	* classifier.
	*
	**/
	void trainingFBinNB(){		
		//1st - calculate pertinences
		calcPertinencesInt();
			
		//defining N
		N = new int[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)
				N[cl,dim] = 100;
		
		//2nd - estimate probability p
		double c1 = 0.1, c2 = 1.0;
		probability = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					probability[cl, dim] += intData[cl,line,dim];
					
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)
				probability[cl, dim] = (c1 + probability[cl, dim]) / (c2 + data.GetLength(1));

		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)
				probability[cl, dim] = probability[cl, dim] / N[cl,dim];	

		/*String matrix = "Probability:\n";				
		for(int i = 0; i < 3; i++){
			matrix += "C" + (i + 1) + "\n";
			for(int j = 0; j < 3; j++){
				matrix += "D" + (j + 1) + ": " + probability[i,j] + "\n";
			}
		}
		
		Debug.Log(matrix);	*/		
	}
	
	/**
	*
	* Method that calculates the density function for the Fuzzy Binomial Naive 
	* Bayes classifier.
	*
	**/
	void assessingFBinNB(){	
		//previously calculates logs in order to reduce running time
		int max = 100;
		for (int i = 0; i < data.GetLength(0); i++)
			for (int j = 0; j < data.GetLength(1); j++)
				for (int k = 0; k < data.GetLength(2); k++)
					if (max < intData[i,j,k])
						max = intData[i,j,k];
					
		double[] logs = new double[max+1];
		for (int j = 2; j < max+1; j++)
			logs[j] = logs[j - 1] + Math.Log(j);
		

		density = new double[data.GetLength(0),(data.GetLength(1)*data.GetLength(0))];
		
		for(int i = 0; i < density.GetLength(0); i++) //3 classe
			for(int j = 0; j < data.GetLength(0); j++) //3 classe
				for(int k = 0; k < data.GetLength(1); k++) //50 linha por classe
					for(int l = 0; l < data.GetLength(2); l++) //3 dimensao
						density[i,((j*data.GetLength(1))+k)] += logs[N[i,l]] - 
												(logs[intData[j,k,l]] + logs[N[i,l] - intData[j,k,l]]) +
												(intData[j,k,l] * Math.Log(probability[i,l])) + ((N[i,l] - intData[j,k,l]) *
												Math.Log(1 - probability[i,l])) + getLogPertinenceInt(i,j,k,l);	
	}
	
	/**
	*
	* Method that estimates parameters for the Fuzzy Poisson Naive Bayes 
	* classifier.
	*
	**/
	void trainingFPoiNB(){		
		//1st - calculate pertinences
		calcPertinencesInt();			
		
		//2nd - calculate mean
		double[,] mean = new double[data.GetLength(0),data.GetLength(2)];		
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					mean[cl, dim] += intData[cl,line,dim];
			
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				mean[cl, dim] /= data.GetLength(1);
		
		//3rd - calculate lambda
		lambda = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					lambda[cl, dim] = mean[cl,dim];			
	}
	
	/**
	*
	* Method that calculates the density function for the Fuzzy Poisson Naive 
	* Bayes classifier.
	*
	**/
	void assessingFPoiNB(){	
		//previously calculates logs in order to reduce running time
		int max = 0;
		for (int i = 0; i < data.GetLength(0); i++)
			for (int j = 0; j < data.GetLength(1); j++)
				for (int k = 0; k < data.GetLength(2); k++)
					if (max < intData[i,j,k])
						max = intData[i,j,k];

		double[] logs = new double[max+1];
		for (int j = 2; j < max+1; j++)
			logs[j] = logs[j - 1] + Math.Log(j);
		

		density = new double[data.GetLength(0),(data.GetLength(1)*data.GetLength(0))];
		
		for(int i = 0; i < density.GetLength(0); i++) //3 classe
			for(int j = 0; j < data.GetLength(0); j++) //3 classe
				for(int k = 0; k < data.GetLength(1); k++) //50 linha por classe
					for(int l = 0; l < data.GetLength(2); l++) //3 dimensao
						density[i,((j*data.GetLength(1))+k)] += (Math.Log(lambda[i,l]) * intData[j,k,l]) - 
																lambda[i,l] - logs[intData[j,k,l]] + 
																getLogPertinenceInt(i,j,k,l);											
				
		/*string text = "";
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 150; j++)
				text += ("density C" + (i+1) + "L" + (j+1) + ": " + density[i,j] + "\n");
		}			
		Debug.Log(text);*/				
	}
	
	/* METHODS FOR CONTINUOS DATA */
	/**
	*
	* Method that converts the data recovered from the text file for continuos 
	* classifiers.
	*
	**/
	void convertToDouble(){
		doubleData = new double[data.GetLength(0), data.GetLength(1), data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int col = 0; col < data.GetLength(2); col++)
					doubleData[cl, line, col] = Convert.ToDouble(data[cl, line, col]);				
	}

	/**
	*
	* Method that calculates the pertinence function for the fuzzy probability.
	* Continuos data only. [Zadeh, 1968]
	*
	**/	
	void calcPertinencesDouble(){
		//1st - calculate sturges
		int sturges = Convert.ToInt32(Math.Ceiling(1 + (3.22 * Math.Log10(doubleData.GetLength(1)))));
		pertinences = new double[doubleData.GetLength(0),sturges,doubleData.GetLength(2),3];

		for (int cl = 0; cl < doubleData.GetLength(0); ++cl) {
			for (int dim = 0; dim < doubleData.GetLength(2); ++dim) {
				//2nd - get max and min
				double max = doubleData[cl,0,dim];
				double min = doubleData[cl,0,dim];
				
				for (int j = 1; j < doubleData.GetLength(1); ++j) {
					if (max < doubleData[cl,j,dim])
						max = doubleData[cl,j,dim];
					if (min > doubleData[cl,j,dim])
						min = doubleData[cl,j,dim];
				}

				//3rd - calculate frequencies
				double[,] freq = new double[sturges,3];
				double step = (max - min) / sturges;

				for (int l = 0; l < sturges; ++l) {
					freq[l,0] = min + step * l;
					freq[l,1] = freq[l,0] + step;
				}

				for(int line = 0; line < doubleData.GetLength(1); ++line){
					for (int st = 0; st < sturges; ++st) {
						if (st == sturges - 1) {
							if (doubleData[cl,line,dim] >= freq[st,0] && doubleData[cl,line,dim] <= freq[st,1])
								freq[st,2] += 1;
						} else if (doubleData[cl,line,dim] >= freq[st,0] && doubleData[cl,line,dim] < freq[st,1]) {
							freq[st,2] += 1;
						}
					}
				}
				
				//4th - calculate pertinences
				double maxFreq = 0;
				for (int st = 1; st < sturges; ++st)
					if (maxFreq < freq[st,2])
						maxFreq = freq[st,2];
					
				for (int st = 0; st < sturges; ++st) {
					pertinences[cl,st,dim,0] = freq[st,0];
					pertinences[cl,st,dim,1] = freq[st,1];
					if (freq[st,2] == 0.0)
						pertinences[cl,st,dim,2] = 0.001;
					else
						pertinences[cl,st,dim,2] = freq[st,2]/maxFreq;
				}
			}
		}
		
		/*String matrix = "Pertinences:\n";				
		for(int i = 0; i < 3; i++){
			matrix += "C" + (i + 1) + "\n";
			for(int j = 0; j < 3; j++){
				matrix += "D" + (j + 1) + ": \n";
				for(int k = 0; k < pertinences.GetLength(1); k++){
				matrix += "[" + pertinences[i,k,j,0] + " - " + pertinences[i,k,j,1] + 
							"]: " + pertinences[i,k,j,2] + "\n";
			}
		}
		}
		
		Debug.Log(matrix);*/
	}
	
	/**
	*
	* Method that returns the pertinence for the specific value from the data.
	* Continuos data only.
	*
	**/	
	double getLogPertinenceDouble(int cl, int auxCl, int line, int dim){
		for (int st = 0; st < pertinences.GetLength(1); ++st) {
			if (st == pertinences.GetLength(1) - 1) {
				if (doubleData[auxCl,line,dim] >= pertinences[cl,st,dim,0] &&
					doubleData[auxCl,line,dim] <= pertinences[cl,st,dim,1])
					return Math.Log(pertinences[cl,st,dim,2]);
			} else if (doubleData[auxCl,line,dim] >= pertinences[cl,st,dim,0] &&
				doubleData[auxCl,line,dim] < pertinences[cl,st,dim,1]){
				return Math.Log(pertinences[cl,st,dim,2]);
			}
		}
		return 0.0;
	}
	
		
	void trainingFExpNB(){
		//1st - calculate pertinences
		calcPertinencesDouble();
		
		//2nd - calculate mean
		double[,] mean = new double[data.GetLength(0),data.GetLength(2)];		
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					mean[cl, dim] += doubleData[cl,line,dim];
			
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				mean[cl, dim] /= data.GetLength(1);
		
		//3rd - calculate lambda
		lambda = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					lambda[cl, dim] = 1/mean[cl,dim];		
	}
	
	
	void assessingFExpNB(){			
		density = new double[data.GetLength(0),(data.GetLength(1)*data.GetLength(0))];
		
		for(int i = 0; i < density.GetLength(0); i++) //3 classe
			for(int j = 0; j < data.GetLength(0); j++) //3 classe
				for(int k = 0; k < data.GetLength(1); k++) //50 linha por classe
					for(int l = 0; l < data.GetLength(2); l++) //3 dimensao
						density[i,((j*data.GetLength(1))+k)] += Math.Log(lambda[i,l]) - 
																lambda[i,l] * doubleData[j,k,l] +
																getLogPertinenceDouble(i,j,k,l);
	}
	
	
	void trainingFGauNB(){
		//1st - calculate pertinences
		calcPertinencesDouble();
		
		//2nd - calculate mean
		mean = new double[data.GetLength(0),data.GetLength(2)];		
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					mean[cl, dim] += doubleData[cl,line,dim];
			
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				mean[cl, dim] /= data.GetLength(1);
		
		//3rd - calculate standard deviation
		std = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					std[cl, dim] += Math.Pow(doubleData[cl,line,dim] - mean[cl,dim], 2.0);		
		
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				std[cl, dim] /= data.GetLength(1);
	}
	
	
	void assessingFGauNB(){			
		density = new double[data.GetLength(0),(data.GetLength(1)*data.GetLength(0))];
		
		for(int i = 0; i < density.GetLength(0); i++) //3 classe
			for(int j = 0; j < data.GetLength(0); j++) //3 classe
				for(int k = 0; k < data.GetLength(1); k++) //50 linha por classe
					for(int l = 0; l < data.GetLength(2); l++) //3 dimensao
						density[i,((j*data.GetLength(1))+k)] += Math.Log(1/std[i,l]) - 
																(Math.Pow(doubleData[j,k,l] - mean[i,l], 2.0)) / 
																(2.0 * Math.Pow(std[i,l], 2.0)) + 
																getLogPertinenceDouble(i,j,k,l);
	}
	
	
	void trainingFGamNB(){
		//1st - calculate pertinences
		calcPertinencesDouble();
		
		//2nd - calculate mean
		mean = new double[data.GetLength(0),data.GetLength(2)];		
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					mean[cl, dim] += doubleData[cl,line,dim];
			
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				mean[cl, dim] /= data.GetLength(1);
				
		//3rd - calculate log of means		
		double[,] logMean = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				logMean[cl, dim] = Math.Log(mean[cl, dim]);
		
		//4th - calculate mean of logs		
		double[,] meanLog = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int line = 0; line < data.GetLength(1); line++)
				for(int dim = 0; dim < data.GetLength(2); dim++)
					meanLog[cl, dim] += Math.Log(doubleData[cl,line,dim]);
				
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				meanLog[cl, dim] /= data.GetLength(1);
		
		//5th - estimate alpha
		alpha = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				alpha[cl,dim] = 0.5 / (logMean[cl,dim] - meanLog[cl,dim]);
			
		//6th - estimate beta
		beta = new double[data.GetLength(0),data.GetLength(2)];
		for(int cl = 0; cl < data.GetLength(0); cl++)
			for(int dim = 0; dim < data.GetLength(2); dim++)	
				beta[cl,dim] = alpha[cl,dim]/mean[cl,dim];
			
		/*String matrix = "Alpha:\n";				
		for(int i = 0; i < 3; i++){
			matrix += "C" + (i + 1) + "\n";
			for(int j = 0; j < 3; j++){
				matrix += "D" + (j + 1) + ": " + alpha[i,j] + "\n";
			}
		}
		
		Debug.Log(matrix);
		
		matrix = "Beta:\n";				
		for(int i = 0; i < 3; i++){
			matrix += "C" + (i + 1) + "\n";
			for(int j = 0; j < 3; j++){
				matrix += "D" + (j + 1) + ": " + beta[i,j] + "\n";
			}
		}
		
		Debug.Log(matrix);	*/	
			
	}
	
	
	void assessingFGamNB(){			
		density = new double[data.GetLength(0),(data.GetLength(1)*data.GetLength(0))];
		
		for(int i = 0; i < density.GetLength(0); i++) //3 classe
			for(int j = 0; j < data.GetLength(0); j++) //3 classe
				for(int k = 0; k < data.GetLength(1); k++) //50 linha por classe
					for(int l = 0; l < data.GetLength(2); l++) //3 dimensao
						density[i,((j*data.GetLength(1))+k)] += 
							alpha[i,l] * Math.Log(beta[i,l]) +
							(alpha[i,l] - 1) * Math.Log(doubleData[j,k,l]) -
							beta[i,l] * doubleData[j,k,l] -
							Math.Log(SpecialFunctions.Gamma(alpha[i,l])) +
							getLogPertinenceDouble(i,j,k,l);						
	}
	
	void assessingWFGamNB(){			
		density = new double[data.GetLength(0),(data.GetLength(1)*data.GetLength(0))];
		
		for(int i = 0; i < density.GetLength(0); i++) //3 classe
			for(int j = 0; j < data.GetLength(0); j++) //3 classe
				for(int k = 0; k < data.GetLength(1); k++) //50 linha por classe
					for(int l = 0; l < data.GetLength(2); l++) //3 dimensao
						density[i,((j*data.GetLength(1))+k)] += 
							weights[i] * (alpha[i,l] * Math.Log(beta[i,l]) +
							(alpha[i,l] - 1) * Math.Log(doubleData[j,k,l]) -
							beta[i,l] * doubleData[j,k,l] -
							Math.Log(SpecialFunctions.Gamma(alpha[i,l])) +
							getLogPertinenceDouble(i,j,k,l));						
	}
	
		
	/* METHODS FOR ALL TYPES OF DATA */
	/**
	*
	* Method that reads the data from the file and saves it as an array of string.
	*
	**/	
	void readDB(){
		string[] lines = System.IO.File.ReadAllLines(@file);
		string[] numbers = lines[0].Split(',');
		
		data = new string[nClass, lines.Length/nClass, numbers.Length-1];
		
		int size = lines.Length/nClass;
		for(int line = 0; line < lines.Length; line++){
			numbers = lines[line].Split(',');		
				for(int dim = 0; dim < numbers.Length-1; dim++)
					data[line/size,line%size,dim] = numbers[dim];			
		}
	}
	
	/**
	*
	* Method that generates and prints the confusion matrix for any method.
	*
	**/
	void assessment(){
		int[,] classMatrix = new int[data.GetLength(0),data.GetLength(0)]; //already starts with 0

		for(int j = 0; j < data.GetLength(1)*data.GetLength(0); j++){
			int right = j/data.GetLength(1);
			 
			if(density[1,j] > density[0,j] && 
			   density[1,j] > density[2,j] )
			   classMatrix[right,1] += 1;
			   
			else if(density[2,j] > density[0,j] &&
					density[2,j] > density[1,j] )
					classMatrix[right,2] += 1;
				
			else classMatrix[right,0] += 1;
		}
		
		String matrix = "\t";
		for(int i = 0; i < data.GetLength(0); i++)
			matrix += ("C" + (i + 1) + "\t");
		matrix += ("\n");
		
		for(int i = 0; i < data.GetLength(0); i++){
			matrix += "C" + (i + 1) + "\t";
			for(int j = 0; j < data.GetLength(0); j++){
				matrix += classMatrix[i,j] + "\t";
			}
			matrix += "\n";
		}
		
		//debug.text = matrix;
		Debug.Log(matrix);
		
		//kappa(classMatrix);
	}

	void kappa(int[,] matrix){
		int nL, nC;
		nL = nC = matrix.GetLength(0); //Numero de linhas e colunas
		int N = 150; //Numero de instancias
		
		/** Calculo do Kappa **/
		int[] sumC = new int[nC]; 
		int[] sumL = new int[nL];
			 
		for(int i = 0; i < nL; i++)
			for(int j = 0; j < nC; j++)
			{
				sumC[i] += matrix[i,j];
				sumL[i] += matrix[j,i];
			}
		
		double pZ = 0;
		double pC = 0;
		
		for(int i = 0; i < nL; i++)
			pZ += matrix[i,i];
		pZ /= N;
		
		for(int i = 0; i < nL; i++)
			pC += sumC[i] * sumL[i];
		pC /= N*N;
		
		double kappa = (pZ - pC)/(1 - pC);
		
		Debug.Log("Kappa: " + kappa + "\n");
		
		/** Calculo da Variancia do Kappa **/		
						
		double t1 = 0.0;
		double t2 = 0.0;		
				
		for(int i = 0; i < nL; i++)
		{
			t1 += matrix[i,i] * (sumC[i] + sumL[i]);	 
			t2 += matrix[i,i] * ((sumC[i] + sumL[i]) * (sumC[i] + sumL[i]));	 
		}
		t1 /= N * N;
		t2 /= N * N * N;
		
		double s1, s2, s3;
		
		double pZC = 1.0 - pZ;
		double pCC = 1.0 - pC;
		
		s1 = pZ * pZC;
		s1 /= N * pCC * pCC;
		
		s2 = 2  * pZC * (2 * pZ * pC - t1);
		s2 /= N * pCC * pCC * pCC;
		
		s3 = pZC * pZC * (t2 - 4 * pC * pC);
		s3 /= N * pCC * pCC * pCC * pCC;
			
		double kappaVar = s1 + s2 + s3;
		
		Debug.Log("Kappa Variance: " + kappaVar + "\n");
		
		/** Calculo da Intervalo de Confianca do Kappa **/
			
		/*double upper = kappa + 1.96 * sqrt(kappaVar); 
		double lower = kappa - 1.96 * sqrt(kappaVar);
		
		cout << "Intervalo de Confianca: (" << upper << ", " << lower << ")" << endl; 
		
		/** Calculo do Teste de Hipoteses do Kappa **/
		/*
		double z = abs(kappa1 - kappa2)/(sqrt(kappaVar1 + kappaVar2));
		
		if(z > 1.96)
			cout << "Ao nivel de 5% de significancia ha diferenca entre os Modelos 1 e 2 de Suporte a Decisao." << endl;
		else
			cout << "Ao nivel de 5% de significancia nao ha diferenca entre os Modelos 1 e 2 de Suporte a Decisao." << endl;
		*/
			
		/** Overall Accuracy Index **/
		
		double oA = pZ*100;
		double oAVar = (pZ * (1 - pZ)) / N;
		
		Debug.Log("Overall Accuracy Index: " + oA + "\n");		
		Debug.Log("Overall Accuracy Variance: " + oAVar + "\n");
		
	}
}