using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class AnamneseScript : MonoBehaviour {

	[SerializeField] Text anamnese; 

	// Use this for initialization
	void Start () {
		anamnese.text = "<b>Idade:</b> x anos \n" +
						"<b>Data do último exame:</b> x meses/anos atrás \n" +
						"<b>Tempo desde a última atividade sexual:</b> x dias/nunca \n" +
						"<b>Histórico de ISTs:</b> - \n" +
						"<b>Data da última mentruação:</b> xx/xx \n" +
						"<b>Queixas:</b> [corrimento, dor, sangramento, dispareunia, etc]";
		/*Fonts: Quiksand-Light, Roboto-Thin, Sofia-Regular*/
	}
	
	// Update is called once per frame
	void Update () {
		
	}
	
	
}
