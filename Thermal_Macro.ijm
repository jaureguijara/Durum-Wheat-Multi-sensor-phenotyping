var input = "$P{batch_input}";
var outputDir = "$P{save_output_dir}";
var saveImages = $P{save_images};
var resultsFile = "$P{save_results_file}";
var removeDarkLine = false;
var kelvin16bitFormat = false;
var threeBands = false;
var list = getFileList(input);

setBatchMode(true);
run("Clear Results");
run("Input/Output...", "jpeg=100 gif=-1 file=.csv use use_file copy_column copy_row save_column save_row");

var fileCount = getImageFileCount(list);
var filenames = newArray(fileCount);
var tPlotMeans = newArray(fileCount);
var tPlotMedians = newArray(fileCount);
var tVegMeans = newArray(fileCount);
var tVegMedians = newArray(fileCount);
var vegetationAreas = newArray(fileCount);
var n = 0;

for (i = 0; i < list.length; i++) {
    if(isImage(list[i])) {
        action(input, list[i]);
    }
}

printResults();

call("java.lang.System.gc");
setBatchMode(false);

function action(input, filename) {
    open(input + filename);
    original = getImageID;
    title = getTitle;
    imageName = File.nameWithoutExtension;

    selectWindow(title);

    //Activate only if images have three bands to remove extra bands from stitch
    if(threeBands) {
        run("Stack to Images");
        selectWindow("Blue");
        close();
        selectWindow("Green");
        close();
        selectWindow("Red");
        rename("TIR");
    }

    //Activate only if the Images are in 16 bit raw Kelvin format
    if(kelvin16bitFormat) {
        run("Grays");
        //Convert image from 16 bit raw to temperature in Kelvin/Celsius
        //Kelvin
        run("Multiply...", "value=0.01");
        //Celsius
        run("Subtract...", "value=273.15");
        run("Enhance Contrast", "saturated=0.35");
    }

    if(removeDarkLine) {
        //Activate only if the Images have a dark line at the bottom (possibly a result of the MosaicTool processing)
        makeRectangle(0, 0, 103, 103);
        run("Crop");
    }

    //This section will divide the image using an automatic thresholding technique, which should give us two groups - vegetation and ground
    rename("image");
    selectWindow("image");
    run("Duplicate...", "title=veg");
    run("Duplicate...", "title=TIR");

    selectWindow("image");
    run("Set Measurements...", "area mean median area_fraction redirect=TIR decimal=4");
    setThreshold(5.0000, 75.0000);
    run("Create Selection");
    run("Measure");
    Tplot_mean = getResult("Mean");
    Tplot_median = getResult("Median");
    
    if(saveImages) {
        selectWindow("image");
        saveAs("JPEG", outputDir + imageName + "_mxEntrpy");
    }

    selectWindow("image");
    run("Convert to Mask");
    rename("mask");
    run("Create Selection");

    //Vegetation Selection
    selectWindow("veg");
    run("Restore Selection");
    setAutoThreshold("Otsu dark");
    run("Convert to Mask");
    run("Restore Selection");
    run("Set Measurements...", "area mean median area_fraction redirect=None decimal=4");
    run("Measure");
    Vegetation_area1 = getResult("Area");

    //Redirect to the original image in order to apply the mask and measure the ground temperature
    selectWindow("veg");
    run("Set Measurements...", "area mean median area_fraction redirect=TIR decimal=4");
    run("Create Selection");
    run("Make Inverse");
    run("Measure");
    Tveg_mean = getResult("Mean");
    Tveg_median = getResult("Median");
    Vegetation_area2 = getResult("Area");

    Vegetation_area = Vegetation_area2 / Vegetation_area1;

    //Clean up after each file
    run("Close All");
    run("Clear Results");

    //Consolidate data
    filenames[n] = filename;
    tPlotMeans[n] = Tplot_mean;
    tPlotMedians[n] = Tplot_median;
    tVegMeans[n] = Tveg_mean;
    tVegMedians[n] = Tveg_median;
    vegetationAreas[n] = Vegetation_area;
    n++;
}

function printResults() {
    run("Clear Results");
    for (i = 0; i < fileCount; i++) {
        setResult("filename", i, filenames[i]);
        setResult("VegetationPercentArea", i, vegetationAreas[i]);
        setResult("Tplot_mean", i, tPlotMeans[i]);
        setResult("Tplot_median", i, tPlotMedians[i]);
        setResult("Tveg_mean", i, tVegMeans[i]);
        setResult("Tveg_median", i, tVegMedians[i]);
    }

    setOption("ShowRowNumbers", false);
    updateResults();
    selectWindow("Results");
    saveAs("Results", resultsFile);
    run("Close"); 
}

function getImageFileCount(list) {
    count = 0;
    for (i = 0; i < list.length; i++) {
        if(isImage(list[i])) {
            count++;
        }
    }
    return count;
}

function isImage(filename) {
    if(endsWith(toLowerCase(filename), ".tif")||
       endsWith(toLowerCase(filename), ".tiff")) {
       return true;
    }
    else {
        return false;
    }
};
