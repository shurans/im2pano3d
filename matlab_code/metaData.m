%metaData
nyu40class = {'wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','refridgerator','television','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop'};
mp40class = {'wall','floor','chair','door','table','picture','cabinet','cushion','window','sofa','bed','curtain','chest_of_drawers','plant','sink','stairs','ceiling','toilet','stool','towel','mirror','tv_monitor','shower','column','bathtub','counter','fireplace','lighting','beam','railing','shelving','blinds','gym_equipment','seating','board_panel','furniture','appliances','clothes','objects','misc'};
mp40To13list = {'wall','floor','chair','door','table','objs','cabinet','objs','window','sofa','bed','window','cabinet','objs','furn','floor','ceiling','objs','chair','objs','objs','tv','furn','wall','furn','cabinet','wall','objs','wall','wall','furn','window','furn','chair','wall','furn','objs','objs','objs','objs'};
nyu40To13list = {'wall','floor','cabinet','bed','chair','sofa','table','door','window','furn','objs','cabinet','window','table','furn','window','cabinet','objs','objs','floor','objs','ceiling','objs','objs','tv','objs','objs','furn','objs','objs','objs','cabinet','objs','furn','objs','furn','objs','wall','furn','objs'};
pano13class = {'ceiling','floor','wall','window','chair','bed','sofa','table','tv','door','cabinet','furn','objs'};
[~,mp40To13_mapId] = ismember(mp40To13list,pano13class);
[~,nyu40To13_mapId] = ismember(nyu40To13list,pano13class);


load('ModelCategoryMapping.mat');


roomTypes = {'Living_Room','Kitchen','Bedroom','Child_Room', ...
             'Dining_Room','Bathroom','Toilet','Hall','Hallway',...
             'Office','Guest_Room','Wardrobe','Room','Lobby','Storage',...
             'Boiler_room','Balcony','Loggia','Terrace',...
             'Entryway','Passenger_elevator','Freight_elevator','Aeration','Garage','Gym'};
roomTypesMapping = {'Living_Room','Kitchen','Bedroom','Bedroom',...
                    'Dining_Room','Bathroom','Bathroom','Hall','Hallway',...
                    'Office','Bedroom','Wardrobe','Room','Hall','Wardrobe','Kitchen','Hallway',...
                    'Hallway','Hallway','Hallway','Room','Room','Room','Garage','Gym'};
                
roomTypesMappingU = {'Living_Room','Kitchen','Bedroom','Dining_Room','Bathroom', 'Hallway','Office','Garage','Gym','Hall','Wardrobe','Room'};
[~,roomTypesMappingId] = ismember(roomTypesMapping,roomTypesMappingU);
nRoomType = 8;


