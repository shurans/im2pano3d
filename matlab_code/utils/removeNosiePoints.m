function inx2remove = removeNosiePoints(points3d,pt)
xmin = prctile(points3d(1,:),pt);
xmax = prctile(points3d(1,:),100-pt);
ymin = prctile(points3d(2,:),pt);
ymax = prctile(points3d(2,:),100-pt);

zmin = prctile(points3d(3,:),pt);
zmax = prctile(points3d(3,:),100-pt);

inx2remove = points3d(1,:)>xmax|points3d(1,:)<xmin|...
             points3d(2,:)>ymax|points3d(2,:)<ymin|...
             points3d(3,:)>zmax|points3d(3,:)<zmin;
end