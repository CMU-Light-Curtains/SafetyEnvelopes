class DefaultViewer {
    constructor(sceneCloud, lcCurtain, logger) {
        this.backend = "http://127.0.0.1:16666";
        this.sceneCloud = sceneCloud;
        this.lcCurtain = lcCurtain;
        this.logger = logger;
        this.int16Factor = 100;

        this.connectStreams();
    }
    connectSceneCloudStream() {
        let self = this;
        var sceneCloudSource = new EventSource(addhttp(self.backend) + "/api/stream_scene_cloud");
        sceneCloudSource.onmessage = function (e) {
            var data = JSON.parse(e.data);

            var points_buf = str2buffer(atob(data["scene_pc_str"]));
            var points = new Int16Array(points_buf);

            self.sceneCloud.update(points, self.int16Factor);
        };
        console.log("Connected to scene cloud stream.");
    }
    connectLcCurtainStream() {
        let self = this;
        var lcCloudSource = new EventSource(addhttp(self.backend) + "/api/stream_lc_curtain");
        lcCloudSource.onmessage = function (e) {
            var data = JSON.parse(e.data);

            const boundaryBuf = str2buffer(atob(data["boundary"]));
            const isectPointsBuf = str2buffer(atob(data["isect_pts"]));

            // boundary
            let intArray = new Int16Array(boundaryBuf);
            const curtainBoundary = new Float32Array(intArray.length);
            for (let i = 0; i < intArray.length; i++)
                curtainBoundary[i] = intArray[i] / self.int16Factor;

            // intersection points
            intArray = new Int16Array(isectPointsBuf);
            const isectPoints = new Float32Array(intArray.length);
            for (let i = 0; i < intArray.length; i++)
                isectPoints[i] = intArray[i] / self.int16Factor;

            self.lcCurtain.update(curtainBoundary, isectPoints);
        };
        console.log("Connected to light curtain cloud stream.");
    }
    connectStreams() {
        let self = this;
        self.connectSceneCloudStream();
        self.connectLcCurtainStream();
    }
}
