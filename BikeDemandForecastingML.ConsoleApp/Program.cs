// This file was auto-generated by ML.NET Model Builder. 

using System;
using BikeDemandForecastingML.Model;

namespace BikeDemandForecastingML.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                IdFeu = 17F,
                Jour = 0F,
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual NbPassant with predicted NbPassant from sample data...\n\n");
            Console.WriteLine($"IdFeu: {sampleData.IdFeu}");
            Console.WriteLine($"Jour: {sampleData.Jour}");
            Console.WriteLine($"\n\nPredicted NbPassant: {predictionResult.Score}\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
