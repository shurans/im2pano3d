function pMap = getPlaneFunction(XYZ,normalMap)
         pMap = sum(XYZ.*normalMap,3);
end