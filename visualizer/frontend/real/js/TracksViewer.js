class TracksViewer {
    constructor(isectTracks, logger) {
        this.backend = "http://127.0.0.1:16666";
        this.isectTracks = isectTracks;
        this.logger = logger;
        this.int16Factor = 100;

        this.connectStreams();
    }
    connectLcCurtainStream() {
        let self = this;
        var lcCloudSource = new EventSource(addhttp(self.backend) + "/api/stream_lc_curtain");
        lcCloudSource.onmessage = function (e) {
            var data = JSON.parse(e.data);
            const isectPointsBuf = str2buffer(atob(data["isect_pts"]));

            // intersection points
            const intArray = new Int16Array(isectPointsBuf);
            const isectPoints = new Float32Array(intArray.length);
            for (let i = 0; i < intArray.length; i++)
                isectPoints[i] = intArray[i] / self.int16Factor;

            self.isectTracks.update(isectPoints);
        };
        console.log("Connected to light curtain cloud stream.");
    }
    connectStreams() {
        let self = this;
        self.connectLcCurtainStream();
    }
}
