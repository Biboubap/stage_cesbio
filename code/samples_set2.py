"""Je souhaite mettre à jour les classes samples et sample_set afin de : 
- les rendre génériques en passant en paramètre du sample_set les chemins vers les rasters ds, dt , dz, et donc en argument des samples
- rendre dt et dz optionnels, s'ils sont à None alors mets toutes les variables qui en découlent à None (mean, var, z_mean). Cela nécessite de changer les autres fonctions de sample et sample set en conséquence. 
Crées-moi deux classes Sample et SamplesSet qui implémentent ces changements."""