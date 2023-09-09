
// W3 & W4: Assignment Part1 : DB Systems (MomgoDB)

// Submitted by SHIVA 

// 1. Open VS Code and connect to MONGODB -- already connected.

// 2. Create a database "ProductOrder" and create collections

// "Product", "Inventory", "User", and  "Order" in it. 

use('ProductOrder');
db.createCollection('Product');
db.createCollection('Inventory');
db.createCollection('User');
db.createCollection('Order');


// 3. Opened MongoDBCompass, navigated to the "ProductOrder" db and 
// imported data to the above  collections 

// 4. Display the first 5 rows of product, inventory, user, and order collection

use ('ProductOrder');
db.Product.find().limit(5);

use ('ProductOrder');
db.Inventory.find().limit(5);

use ('ProductOrder');
db.User.find().limit(5);

use ('ProductOrder');
db.Order.find().limit(5);


// 5. Display the Unique Brand and Model names from the Product collection
// use ('ProductOrder');

use ('ProductOrder');
db.Product.distinct('brand'); //unique brands 

use ('ProductOrder');
db.Product.distinct('model'); //unique models 


// 6. Find the maximum and minimum price of the given products
// use ('ProductOrder');

use ('ProductOrder');
db.Product.aggregate([{"$group": {
                "_id": null,
                "Max Price": {"$max":"$price"},
                "Min Price": {"$min":"$price"}
                }}]);

// 7. Display the quantity and last_updated date and time for sku "SNY-11001"
// use ('ProductOrder');
use ('ProductOrder');
db.Inventory.find(
    {"sku" : 'SNY-11001'},
    {"_id":0,"sku":1,"quantity": 1,"last_updated": 1});

// 8. List down the count of the total number of users whose role is identified as
//'Supplier' from User collection
// use ('ProductOrder');
use ('ProductOrder');
db.User.find({role:"Supplier"}).count();

// 9. Display 'sku', 'code', 'price', 'brand' and 'warranty' information for the model
// 'Bravia-X'
// use ('ProductOrder');
use ('ProductOrder');
db.Product.find({model:"Bravia-X"},
                {"_id":0,"sku":1,"code":1,"price":1,"brand":1,"warranty":1})

// 10. Find all the information of Sony products which have an Price greater
// than 1 lakh
// use ('ProductOrder');
use ('ProductOrder');
db.Product.find({$and:[{brand:"Sony"},{price:{$gt:100000}}]})

// 11. Find the total no of products by each Brand and sort them in descending order
// use ('ProductOrder');
use ('ProductOrder');
db.Product.aggregate([{$group: { _id:"$brand", count: { $sum: 1}}},
    {$sort:{'count':-1}}
    ])

// 12. Find the total no of users by each role, sort them is descending order and
// save the results in the temporary collection 
// use ('ProductOrder');
use ('ProductOrder');
db.User.aggregate([
    {$match:{role:{$ne:null}}}, 
    {$group:{_id:{role:'$role',listed_in:'$listed_in'}, no_of_users:{$sum:1}}}, 
    {$sort:{no_of_users: -1}}, // 1 = asc, -1 = desc
    {$out: "temp_users_by_role"}
]);
db.temp_users_by_role.find().pretty()

