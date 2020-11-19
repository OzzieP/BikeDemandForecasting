// This file was auto-generated by ML.NET Model Builder. 

using Microsoft.ML.Data;

namespace BikeDemandForecastingML.Model
{
    public class ModelInput
    {
        [ColumnName("idEtat"), LoadColumn(0)]
        public float IdEtat { get; set; }


        [ColumnName("idFeu"), LoadColumn(1)]
        public float IdFeu { get; set; }


        [ColumnName("jour"), LoadColumn(2)]
        public float Jour { get; set; }


        [ColumnName("horaire"), LoadColumn(3)]
        public string Horaire { get; set; }


        [ColumnName("nbPassant"), LoadColumn(4)]
        public float NbPassant { get; set; }


        [ColumnName("etat"), LoadColumn(5)]
        public bool Etat { get; set; }


    }
}