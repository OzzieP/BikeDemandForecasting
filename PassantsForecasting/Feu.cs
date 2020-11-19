using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Workshop.Models
{
    public class Feu
    {
        public int idFeu { get; set; }
        public string matricule { get; set; }

        public Feu()
        {

        }

        public Feu(int idFeu, string matricule)
        {
            this.idFeu = idFeu;
            this.matricule = matricule;
        }

        public Feu(string matricule)
        {
            this.matricule = matricule;
        }
    }
}