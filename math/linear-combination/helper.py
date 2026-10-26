class Finance:
    def _flatten(self, args):
        flat_list = []
        for item in args:
            if isinstance(item, (list, tuple)):
                flat_list.extend(self._flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    def total(self, *args):
        data = self._flatten(args)
        return sum(data)

    def average(self, *args):
        data = self._flatten(args)
        if not data:
            return 0.0
        return sum(data) / len(data)
